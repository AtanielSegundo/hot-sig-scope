/*
 * ============================================================================
 *  ESP32 - Filtro FIR em tempo real  (ADC -> convolucao rapida FFT -> DAC)
 *  Arduino IDE + FreeRTOS  |  2 nucleos  |  Fs = 16 kHz  (T = 62.5 us)
 * ----------------------------------------------------------------------------
 *  Mesma cadeia do filtro IIR (timer 16 kHz -> aquisicao -> processamento ->
 *  DAC), mas a ProcessTask aplica um filtro FIR por CONVOLUCAO RAPIDA
 *  (overlap-save):
 *
 *     bloco de B amostras novas  ->  buf = [ hist(N-1) | B novas ]  (= N_FFT)
 *     X = FFT(buf);   Y = X .* H;   y = IFFT(Y)
 *     descarta as N-1 primeiras (alias da convolucao circular) e usa as
 *     ultimas B amostras  ->  saida linear correta do filtro
 *
 *  Os coeficientes h[n], N (taps), M (atraso) e N_FFT vem do cabecalho
 *  fir_taps.h, gerado pelo scope (core.fir_core.write_taps_header). O ESP32
 *  faz H = FFT(h zero-padded) UMA vez no setup().
 *
 *  B = N_FFT - (N-1)  amostras uteis por bloco  ->  entra B, sai B (mesma
 *  taxa), entao todo o pipeline (timer / ring / fila) e identico ao do IIR.
 *  Latencia de bloco ~ B / Fs  (menor N_FFT -> menor latencia).
 *
 *  Modelo de sinal (igual ao IIR):
 *    Entrada x(t) = Vref/2 + f(t);  filtra-se f(t) -> g(t);  Saida y = Vref/2 + g
 *
 *  REQUER o componente ESP-DSP (esp_dsp.h) para a FFT radix-2 fc32.
 * ============================================================================
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "esp_dsp.h"

// FIR_FS_HZ, FIR_N_TAP, FIR_M, FIR_N_FFT, FIR_B_BLK, h_taps[]
#include "fir_taps.h"   

#define SAMPLE_RATE   16000             // Hz - deve casar com FIR_FS_HZ
#define BLOCK_SIZE    FIR_B_BLK         // amostras uteis por bloco (overlap-save)
#define ADC_PIN       34                // GPIO34 = ADC1_CH6 (entrada, input-only)
#define DAC_PIN       25                // GPIO25 = DAC1     (saida)

#define VREF          (3.3f)
#define ADC_FS        (4095.0f)      
#define DAC_FS        (255.0f)       
#define MID           (VREF * 0.5f)  

#define ADC_RING_LEN  (4 * FIR_N_FFT)   // buffer circular: folga p/ varios blocos B

static float Hspec [2 * FIR_N_FFT];     // H = FFT(h zero-padded ate N_FFT)
static float fftbuf[2 * FIR_N_FFT];     // espectro de trabalho do bloco
static float hist  [FIR_N_TAP - 1];     // estado overlap-save: N-1 entradas anteriores

static volatile uint16_t adcRing[ADC_RING_LEN];
static volatile uint32_t adcHead = 0;   // somente a AcquireTask escreve
static volatile uint32_t adcTail = 0;   // somente a ProcessTask escreve

typedef struct { uint8_t s[BLOCK_SIZE]; } Block;

static QueueHandle_t     dacQueue   = NULL;  // blocos prontos p/ o DAC
static SemaphoreHandle_t adcSem     = NULL;  // "tick" de aquisicao (binario)
static SemaphoreHandle_t dacSem     = NULL;  // "tick" de saida     (binario)
static SemaphoreHandle_t blockReady = NULL;  // sinaliza bloco completo no ring

static hw_timer_t *timer = NULL;

void IRAM_ATTR onTimer() {
  BaseType_t hpw = pdFALSE;
  xSemaphoreGiveFromISR(adcSem, &hpw);   // libera 1 amostra de entrada
  xSemaphoreGiveFromISR(dacSem, &hpw);   // libera 1 amostra de saida
  portYIELD_FROM_ISR(hpw);
}

// ------------------------- FFT complexa (ESP-DSP) ---------------------------
//  d: N complexos intercalados (re,im), in-place. IFFT via conjugacao:
//      ifft(X) = conj( fft( conj(X) ) ) / N
static void fft_run(float *d, int n, bool inverse) {
  if (inverse) for (int i = 0; i < n; i++) d[2 * i + 1] = -d[2 * i + 1];  // conj entrada
  dsps_fft2r_fc32(d, n);
  dsps_bit_rev_fc32(d, n);            // (em algumas versoes do ESP-DSP: dsps_bit_rev2r_fc32)
  if (inverse) {
    float s = 1.0f / n;
    for (int i = 0; i < n; i++) { d[2 * i] *= s; d[2 * i + 1] *= -s; }      // conj + 1/N
  }
}

// --------------- Tarefa 1: aquisicao (ADC -> buffer circular) ---------------
void AcquireTask(void *arg) {
  uint16_t count = 0;
  for (;;) {
    xSemaphoreTake(adcSem, portMAX_DELAY);     // espera o tick T
    uint16_t raw = analogRead(ADC_PIN);        // 0..4095

    uint32_t next = (adcHead + 1) % ADC_RING_LEN;
    if (next != adcTail) {                      // ring nao esta cheio
      adcRing[adcHead] = raw;
      adcHead = next;
      if (++count >= BLOCK_SIZE) {             // completou um bloco de B amostras?
        count = 0;
        xSemaphoreGive(blockReady);            // avisa a ProcessTask
      }
    }
  }
}

// ------------- Tarefa 2: processamento (FIR por overlap-save) ---------------
void ProcessTask(void *arg) {
  Block blk;

  for (;;) {
    xSemaphoreTake(blockReady, portMAX_DELAY);  // espera 1 bloco de B amostras

    // monta o buffer: [ hist(N-1) | B amostras novas ] = N_FFT  (imag = 0)
    for (int i = 0; i < FIR_N_TAP - 1; i++) { fftbuf[2 * i] = hist[i]; fftbuf[2 * i + 1] = 0.0f; }
    for (int i = 0; i < FIR_B_BLK; i++) {
      uint16_t raw = adcRing[adcTail];
      adcTail = (adcTail + 1) % ADC_RING_LEN;

      float f = raw * (VREF / ADC_FS) - MID;    // conta ADC -> tensao -> remove DC -> f(t)
      int p = (FIR_N_TAP - 1) + i;
      fftbuf[2 * p] = f; fftbuf[2 * p + 1] = 0.0f;
    }

    // estado p/ o proximo bloco = ultimas N-1 entradas (LER antes da FFT in-place)
    for (int i = 0; i < FIR_N_TAP - 1; i++) hist[i] = fftbuf[2 * (FIR_B_BLK + i)];

    // ----- convolucao rapida:  X = FFT(buf);  Y = X .* H;  y = IFFT(Y) -----
    fft_run(fftbuf, FIR_N_FFT, false);
    for (int k = 0; k < FIR_N_FFT; k++) {
      float xr = fftbuf[2 * k],     xi = fftbuf[2 * k + 1];
      float hr = Hspec[2 * k],      hi = Hspec[2 * k + 1];
      fftbuf[2 * k]     = xr * hr - xi * hi;
      fftbuf[2 * k + 1] = xr * hi + xi * hr;
    }
    fft_run(fftbuf, FIR_N_FFT, true);

    // descarta as N-1 primeiras (alias circular) e usa as ultimas B amostras
    for (int i = 0; i < FIR_B_BLK; i++) {
      float g  = fftbuf[2 * ((FIR_N_TAP - 1) + i)];   // saida linear correta
      float yv = MID + g;                             // reconstroi y(t)=Vref/2+g(t)

      int code = (int)lroundf(yv * (DAC_FS / VREF));  // tensao -> conta DAC
      if (code < 0)   code = 0;                       // saturacao (evita wrap)
      if (code > 255) code = 255;
      blk.s[i] = (uint8_t)code;
    }

    xQueueSend(dacQueue, &blk, portMAX_DELAY);  // envia bloco p/ a DacTask
  }
}

// ------------------- Tarefa 3: saida (DAC, 1 amostra / T) -------------------
void DacTask(void *arg) {
  Block blk;
  for (;;) {
    if (xQueueReceive(dacQueue, &blk, portMAX_DELAY) == pdTRUE) {
      for (int i = 0; i < BLOCK_SIZE; i++) {
        xSemaphoreTake(dacSem, portMAX_DELAY);  // espera o tick T
        dacWrite(DAC_PIN, blk.s[i]);            // emite UMA amostra
      }
    }
  }
}

// --------------------------------- setup ------------------------------------
void setup() {
  Serial.begin(115200);
  delay(200);

  // ADC: 12 bits e atenuacao para abranger ~0..Vref
  analogReadResolution(12);
  analogSetPinAttenuation(ADC_PIN, ADC_11db);

  // FFT (ESP-DSP): tabela de twiddles + H = FFT(h zero-padded ate N_FFT), uma vez
  if (dsps_fft2r_init_fc32(NULL, FIR_N_FFT) != ESP_OK) {
    Serial.println("Falha ao iniciar a FFT (ESP-DSP). Verifique o componente esp-dsp.");
    while (true) vTaskDelay(pdMS_TO_TICKS(1000));
  }
  for (int i = 0; i < FIR_N_FFT; i++) {
    Hspec[2 * i]     = (i < FIR_N_TAP) ? h_taps[i] : 0.0f;   // h zero-padded
    Hspec[2 * i + 1] = 0.0f;
  }
  fft_run(Hspec, FIR_N_FFT, false);
  for (int i = 0; i < FIR_N_TAP - 1; i++) hist[i] = 0.0f;

  // Recursos do FreeRTOS
  adcSem     = xSemaphoreCreateBinary();
  dacSem     = xSemaphoreCreateBinary();
  blockReady = xSemaphoreCreateCounting(ADC_RING_LEN / BLOCK_SIZE, 0);
  dacQueue   = xQueueCreate(8, sizeof(Block));

  // Tarefas: I/O no nucleo 0, DSP (FFT) no nucleo 1  (uso dos 2 nucleos)
  xTaskCreatePinnedToCore(AcquireTask, "ADC", 4096, NULL, 5, NULL, 0);
  xTaskCreatePinnedToCore(ProcessTask, "DSP", 8192, NULL, 4, NULL, 1);
  xTaskCreatePinnedToCore(DacTask,     "DAC", 4096, NULL, 5, NULL, 0);

  // Timer mestre 16 kHz (base 2 MHz, alarme a cada 125 ticks => 62.5 us)
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  timer = timerBegin(2000000);                  // core 3.x: resolucao 2 MHz
  timerAttachInterrupt(timer, &onTimer);
  timerAlarm(timer, 2000000 / SAMPLE_RATE, true, 0);
#else
  timer = timerBegin(0, 40, true);              // core 2.x: 80 MHz / 40 = 2 MHz
  timerAttachInterrupt(timer, &onTimer, true);
  timerAlarmWrite(timer, 2000000 / SAMPLE_RATE, true);
  timerAlarmEnable(timer);
#endif

  Serial.printf("Filtro FIR (overlap-save) iniciado: N=%d taps, N_FFT=%d, B=%d, Fs=%.0f Hz\n",
                FIR_N_TAP, FIR_N_FFT, FIR_B_BLK, (double)FIR_FS_HZ);
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}
