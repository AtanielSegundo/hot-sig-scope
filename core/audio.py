import numpy as np
import raylib as rl


class ToneStreamer:
    '''
    Click-free playback of the bench's test tone through the speaker, using
    raylib's own audio device (no extra dependency).

    The bench shows a single swept sinusoid at `f_now` whose amplitude is the
    filter response |H(f_now)|. Instead of re-sending the windowed display buffer
    every frame (which would click at the seams), this keeps a phase accumulator
    so the sine stays continuous across frames while frequency / amplitude track
    the sweep in real time. Optional amplitude quantization reproduces, audibly,
    the same bit depth shown in the time/FFT panel (quantization distortion).
    '''
    def __init__(self, sample_rate=44100, frames=2048, channels=1):
        if not rl.IsAudioDeviceReady():
            rl.InitAudioDevice()
        rl.SetAudioStreamBufferSizeDefault(int(frames))
        self.fs      = int(sample_rate)
        self.frames  = int(frames)
        self.stream  = rl.LoadAudioStream(self.fs, 16, int(channels))  # 16-bit PCM
        self.phase   = 0.0
        self.playing = False

    @staticmethod
    def _quantize(x, bits):
        '''Mid-rise quantization over full scale [-1, 1] (mirrors render.quantize).'''
        levels = 1 << int(bits)
        step   = 2.0 / levels
        q      = np.clip(np.floor((x + 1.0) / step), 0, levels - 1)
        return -1.0 + (q + 0.5) * step

    def feed(self, freq, amp=1.0, bits=None, volume=0.4):
        '''
        Keep the stream fed with a continuous tone at `freq` (Hz), scaled by `amp`.
        Call once per frame; it tops up every buffer the device has drained.
        '''
        if not self.playing:
            rl.PlayAudioStream(self.stream)
            self.playing = True
        rl.SetAudioStreamVolume(self.stream, float(volume))

        dphi = 2.0 * np.pi * float(freq) / self.fs
        k    = np.arange(self.frames)
        while rl.IsAudioStreamProcessed(self.stream):
            x = float(amp) * np.sin(self.phase + dphi * k)
            self.phase = (self.phase + dphi * self.frames) % (2.0 * np.pi)
            x = np.clip(x, -1.0, 1.0)
            if bits is not None:
                x = self._quantize(x, bits)
            pcm = (x * 32767.0).astype('<i2')                  # int16, little-endian
            ptr = rl.ffi.cast("void *", rl.ffi.from_buffer(pcm))
            rl.UpdateAudioStream(self.stream, ptr, self.frames)

    def stop(self):
        if self.playing:
            rl.PauseAudioStream(self.stream)
            self.playing = False

    def close(self):
        try:
            rl.UnloadAudioStream(self.stream)
        except Exception:
            pass
