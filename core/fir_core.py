import numpy as np
import math
import os

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


def write_taps_header(path, h, N, M, n_fft, fs,
                      window=None, low_cut=None, high_cut=None, bw=None,
                      per_line=4):
    '''
    Emit a C header (fir_taps.h) describing a FIR design so an Arduino / ESP32
    sketch can #include it at compile time (used by filtro_fir_esp32.ino).

    Only the N REAL taps are written (any zero padding in `h` is dropped); the
    device zero-pads to n_fft and computes H = FFT(h) once at boot, then filters
    by overlap-save. The header defines:

        FIR_FS_HZ   sample rate (float)
        FIR_N_TAP   N  -- real taps (odd -> exact linear phase)
        FIR_M       M  -- group delay in samples
        FIR_N_FFT   n_fft -- FFT length (power of two)
        FIR_B_BLK   usable block per FFT = n_fft - (N-1)
        h_taps[FIR_N_TAP]   the coefficients (float, with 'f' suffix)

    n_fft must be a power of two and > N (so B = n_fft-(N-1) >= 1).
    Returns the absolute path written.
    '''
    N  = int(N)
    nf = int(n_fft)
    if nf & (nf - 1):
        raise ValueError(f"N_FFT={nf} must be a power of two (radix-2 FFT)")
    if nf <= N:
        raise ValueError(f"N_FFT={nf} must be > N={N} (block B = N_FFT-(N-1) must be >= 1)")

    taps = np.asarray(h, dtype=float).ravel()[:N]      # only the real taps
    B    = nf - (N - 1)
    win  = getattr(window, "__name__", window)

    desc = []
    if win is not None:
        desc.append(str(win))
    if low_cut is not None and high_cut is not None:
        desc.append(f"band {low_cut:g}..{high_cut:g} Hz")
    if bw is not None:
        desc.append(f"bw={bw:g} Hz")
    desc.append(f"@ {fs:g} Hz")

    out = []
    out.append("// AUTO-GENERATED by scope core.fir_core.write_taps_header -- DO NOT EDIT BY HAND")
    out.append(f"// FIR design: {'  '.join(desc)}")
    out.append(f"// N={N} taps, group delay M={M} samples ({M / float(fs) * 1e3:.3f} ms)")
    out.append(f"// overlap-save: N_FFT={nf}, block B=N_FFT-(N-1)={B} "
               f"(block latency {B / float(fs) * 1e3:.2f} ms)")
    out.append("#ifndef FIR_TAPS_H")
    out.append("#define FIR_TAPS_H")
    out.append("")
    out.append(f"#define FIR_FS_HZ   {float(fs):.1f}f")
    out.append(f"#define FIR_N_TAP   {N}")
    out.append(f"#define FIR_M       {int(M)}")
    out.append(f"#define FIR_N_FFT   {nf}")
    out.append(f"#define FIR_B_BLK   (FIR_N_FFT - (FIR_N_TAP - 1))   // = {B}")
    out.append("")
    out.append("static const float h_taps[FIR_N_TAP] = {")
    for i in range(0, N, per_line):
        row = "  " + " ".join(f"{v: .9e}f," for v in taps[i:i + per_line])
        out.append(row)
    out.append("};")
    out.append("")
    out.append("#endif // FIR_TAPS_H")
    text = "\n".join(out) + "\n"

    path = os.path.abspath(path)
    d    = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write(text)
    return path