/*
 * ============================================================================
 *  ESP32 - Filtro IIR em tempo real  (ADC -> equacao de diferenca -> DAC)
 *  Arduino IDE + FreeRTOS  |  2 nucleos  |  Fs = 16 kHz  (T = 62.5 us)
 * ----------------------------------------------------------------------------
 *  Cadeia de processamento:
 *
 *    Timer 16 kHz (ISR)  ->  libera "ticks" (semaforos binarios) a cada T
 *           |
 *           |-- AcquireTask  (nucleo 0): a cada tick le o ADC e grava no
 *           |                            buffer circular (amostragem continua)
 *           |
 *    ProcessTask (nucleo 1): le blocos do buffer circular, remove o offset
 *           |                Vref/2, aplica a equacao de diferenca e
 *           |                reconstroi y(t)=Vref/2+g(t); envia o bloco
 *           |                pronto para a fila do DAC
 *           |
 *    DacTask (nucleo 0): recebe blocos prontos e, a cada tick (T),
 *                        escreve UMA amostra no DAC
 *
 *  Modelo de sinal:
 *    Entrada: x(t) = Vref/2 + f(t),  com x_max = Vref = 3.3 V
 *    Filtra-se SOMENTE f(t) (componente AC, sem o nivel DC) -> g(t)
 *    Saida:   y(t) = Vref/2 + g(t)
 *
 *  Equacao de diferenca (biquad IIR de 2a ordem):
 *    y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2] + a1*y[k-1] + a2*y[k-2]
 *  (aqui x = f, y = g)
 * ============================================================================
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

// ------------------------------ Configuracao --------------------------------
#define SAMPLE_RATE   16000          // Hz  (T = 62.5 us)
#define BLOCK_SIZE    16             // amostras por bloco (latencia = BLOCK_SIZE / SAMPLE_RATE)
#define ADC_PIN       34             // GPIO34 = ADC1_CH6 (entrada, input-only)
#define DAC_PIN       25             // GPIO25 = DAC1     (saida)

#define VREF          (3.3f)
#define ADC_FS        (4095.0f)      // fundo de escala do ADC (12 bits)
#define DAC_FS        (255.0f)       // fundo de escala do DAC (8 bits)
#define MID           (VREF * 0.5f)  // Vref/2 = offset de 1.65 V

#define ADC_RING_LEN  1024           // tamanho do buffer circular do ADC

// ------------------- Coeficientes da equacao de diferenca -------------------
//  y[k] = b0*x[k] + b1*x[k-1] + b2*x[k-2] + a1*y[k-1] + a2*y[k-2]
static const float b0 =  0.0986102f;
static const float b1 = -0.196828f;
static const float b2 =  0.0986023f;
static const float a1 =  1.96995f; 
static const float a2 = -0.970447f;

// -------------------------------- Recursos ----------------------------------
static volatile uint16_t adcRing[ADC_RING_LEN];
static volatile uint32_t adcHead = 0;   // somente a AcquireTask escreve
static volatile uint32_t adcTail = 0;   // somente a ProcessTask escreve

typedef struct { uint8_t s[BLOCK_SIZE]; } Block;

static QueueHandle_t     dacQueue   = NULL;  // blocos prontos p/ o DAC
static SemaphoreHandle_t adcSem     = NULL;  // "tick" de aquisicao (binario)
static SemaphoreHandle_t dacSem     = NULL;  // "tick" de saida     (binario)
static SemaphoreHandle_t blockReady = NULL;  // sinaliza bloco completo no ring

static hw_timer_t *timer = NULL;

// ------------------------ ISR do timer mestre (16 kHz) ----------------------
//  Mantida minima e IRAM-safe: apenas libera os dois "ticks".
void IRAM_ATTR onTimer() {
  BaseType_t hpw = pdFALSE;
  xSemaphoreGiveFromISR(adcSem, &hpw);   // libera 1 amostra de entrada
  xSemaphoreGiveFromISR(dacSem, &hpw);   // libera 1 amostra de saida
  portYIELD_FROM_ISR(hpw);
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
      if (++count >= BLOCK_SIZE) {             // completou um bloco?
        count = 0;
        xSemaphoreGive(blockReady);            // avisa a ProcessTask
      }
    }
  }
}

// ------------- Tarefa 2: processamento (equacao de diferenca) ---------------
void ProcessTask(void *arg) {
  // Estado do filtro mantido ENTRE blocos (continuidade da convolucao):
  float x1 = 0, x2 = 0;   // f[k-1], f[k-2]
  float y1 = 0, y2 = 0;   // g[k-1], g[k-2]
  Block blk;

  for (;;) {
    xSemaphoreTake(blockReady, portMAX_DELAY); // espera 1 bloco completo

    for (int i = 0; i < BLOCK_SIZE; i++) {
      uint16_t raw = adcRing[adcTail];
      adcTail = (adcTail + 1) % ADC_RING_LEN;

      float xv = raw * (VREF / ADC_FS);          // conta ADC -> tensao
      float f  = xv - MID;                       // remove offset -> f(t)

      // ----- equacao de diferenca -----
      float g  = b0 * f + b1 * x1 + b2 * x2 + a1 * y1 + a2 * y2;

      // atualiza historico
      x2 = x1; x1 = f;
      y2 = y1; y1 = g;

      float yv = MID + g;                        // reconstroi y(t)=Vref/2+g(t)

      int code = (int)lroundf(yv * (DAC_FS / VREF)); // tensao -> conta DAC
      if (code < 0)   code = 0;                  // saturacao (evita wrap)
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

  // Recursos do FreeRTOS
  adcSem     = xSemaphoreCreateBinary();
  dacSem     = xSemaphoreCreateBinary();
  blockReady = xSemaphoreCreateCounting(ADC_RING_LEN / BLOCK_SIZE, 0);
  dacQueue   = xQueueCreate(8, sizeof(Block));

  // Tarefas: I/O no nucleo 0, DSP no nucleo 1  (uso dos 2 nucleos)
  xTaskCreatePinnedToCore(AcquireTask, "ADC", 4096, NULL, 5, NULL, 0);
  xTaskCreatePinnedToCore(ProcessTask, "DSP", 4096, NULL, 4, NULL, 1);
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

  Serial.println("Filtro IIR em tempo real iniciado (Fs = 16 kHz).");
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000));
}
