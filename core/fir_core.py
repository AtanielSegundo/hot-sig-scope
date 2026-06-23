import numpy as np
import math

class Window:
    @staticmethod
    def window(n, N):
        pass
    @staticmethod
    def rule(df,fs):
        pass

class Hamming(Window):
    @staticmethod
    def window(n, N):
        return 0.54 - 0.46 * math.cos((2.0 * math.pi * n) / (N - 1))
    @staticmethod
    def rule(df, fs):
        return math.ceil(3.3 / (df / fs))

class Hanning(Window):    
    @staticmethod
    def window(n, N):
        return 0.5 - 0.5 * math.cos((2.0 * math.pi * n) / (N - 1))
    @staticmethod
    def rule(df, fs):
        return math.ceil(3.1 / (df / fs))

class Rectangular(Window):
    @staticmethod
    def window(n, N):
        return 1.0 if 0 <= n < N else 0.0
    @staticmethod
    def rule(df, fs):
        return math.ceil(0.9 / (df / fs))

class Blackman(Window):
    @staticmethod
    def window(n, N):
        return (0.42 - 0.5 * math.cos((2.0 * math.pi * n) / (N - 1)) +
                0.08 * math.cos((4.0 * math.pi * n) / (N - 1)))
    @staticmethod
    def rule(df, fs):
        return math.ceil(5.5 / (df / fs))

def fir_taps(w:Window,
             fs:float,
             bw:float,
             low_cut:float,
             high_cut:float,
             length=None):
    '''
    Design a windowed-sinc FIR (band / low / high-pass set by the cuts) and
    return (h, N, M):

        h   impulse response / filter coefficients
        N   number of REAL taps -- forced ODD so the filter is exactly linear phase
        M   group delay in samples, M = (N-1)/2

    The tap count comes from the window's transition rule  N = w.rule(bw, fs):
    a narrower transition bandwidth bw -> more taps -> sharper roll-off.

    `length` lets you fix the size of the returned array. The N designed taps are
    placed at the front and the rest is ZERO-PADDED, so h has exactly `length`
    samples while N / M still describe the real filter. Padding goes after the
    taps, so it changes nothing about the response (same magnitude AND phase) --
    it only fixes the buffer length and makes any later FFT's bins finer. Must be
    >= N (a shorter length would truncate the filter); raises ValueError if not.
    '''
    N = w.rule(bw, fs)
    if N % 2 == 0:
        N += 1
    M = (N - 1) // 2

    L = N if length is None else int(length)
    if L < N:
        raise ValueError(f"length={L} < required tap count N={N}; "
                         f"increase length or widen bw (bigger bw -> fewer taps)")

    h = np.zeros(L)                                          # trailing samples stay 0
    for n in range(N):
        m = n - M
        if m == 0:
            h[n] = 2 * (high_cut - low_cut) / fs            # ideal sinc value at center
        else:
            h[n] = (math.sin(2 * math.pi * high_cut * m / fs) -
                    math.sin(2 * math.pi * low_cut * m / fs)) / (math.pi * m)
        h[n] *= w.window(n, N)                              # window over the REAL N taps

    return h, N, M


def filter_freq_response(w:Window,
                         fs:float,
                         bw:float,
                         low_cut:float,
                         high_cut:float,
                         freq_points,
                         nfft=None):

    h, N, M = fir_taps(w, fs, bw, low_cut, high_cut)

    if nfft is None:
        # oversample the next power of two >= N so the FFT bins are much
        # finer than the spacing of freq_points (smoother true response)
        nfft = (1 << (N - 1).bit_length()) * 16
    nfft = max(nfft, N)  # never truncate the impulse response

    H = np.fft.rfft(h, n=nfft)
    freq_h = np.fft.rfftfreq(nfft, d=1/fs)
    H_mag = np.interp(freq_points, freq_h, np.abs(H))

    return H_mag