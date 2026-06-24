import numpy as np
import fir_core as fir

# Band-pass design under test: pass 800..1500 Hz at fs = 8000 Hz.
FS       = 8000.0
LOW_CUT  = 800.0
HIGH_CUT = 1500.0
BW       = 200.0        # transition bandwidth (sets the tap count)
WINDOW   = fir.Hamming
NSAMP    = 8000         # 1 s -> 1 Hz FFT bins, so integer-Hz tones land on a bin


def tone(freq, fs=FS, n=NSAMP, amp=1.0):
    """A pure cosine at `freq` Hz."""
    k = np.arange(n)
    return amp * np.cos(2.0 * np.pi * freq * k / fs)


def apply_filter(signal, fs=FS, nfft=None):
    """Filter `signal` by multiplying its spectrum with the FIR response.

    Returns (freqs, X, H, Y, y) where:
      X = signal spectrum, H = filter magnitude on the same bins,
      Y = X * H = filtered spectrum, y = time-domain filtered signal.
    """
    X = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    H = fir.filter_freq_response(WINDOW, fs, BW, LOW_CUT, HIGH_CUT, freqs, nfft=nfft)
    Y = X * H
    y = np.fft.irfft(Y, n=len(signal))
    return freqs, X, H, Y, y


def bin_amp(spectrum, freqs, freq):
    """Peak amplitude carried by the bin nearest `freq`."""
    k = int(np.argmin(np.abs(freqs - freq)))
    return 2.0 * np.abs(spectrum[k]) / NSAMP


def test_single_tones():
    """Each pure tone in -> measure how much comes out (gain = out/in)."""
    print("=== single pure tones (pass band 800..1500 Hz) ===")
    print(f"{'freq Hz':>8} {'in amp':>8} {'out amp':>8} {'gain':>7}  region")
    cases = [
        (100, "stop (below)"),
        (300, "stop (below)"),
        (800, "transition (low cut)"),
        (1000, "PASS"),
        (1150, "PASS"),
        (1500, "transition (high cut)"),
        (2500, "stop (above)"),
        (3500, "stop (above)"),
    ]
    for f, region in cases:
        x = tone(f)
        freqs, X, H, Y, y = apply_filter(x)
        a_in = bin_amp(X, freqs, f)
        a_out = bin_amp(Y, freqs, f)
        gain = a_out / a_in if a_in else 0.0
        print(f"{f:>8} {a_in:>8.3f} {a_out:>8.3f} {gain:>7.3f}  {region}")


def test_multitone():
    """Sum of three tones -> only the in-band one should survive."""
    print("\n=== composite signal: 300 + 1000 + 3000 Hz ===")
    x = tone(300) + tone(1000) + tone(3000)
    freqs, X, H, Y, y = apply_filter(x)
    print(f"{'freq Hz':>8} {'in amp':>8} {'out amp':>8} {'gain':>7}")
    for f in (300, 1000, 3000):
        a_in = bin_amp(X, freqs, f)
        a_out = bin_amp(Y, freqs, f)
        print(f"{f:>8} {a_in:>8.3f} {a_out:>8.3f} {a_out / a_in:>7.3f}")
    # crude time-domain check: filtered output amplitude ~ the 1 kHz tone alone
    print(f"output peak |y| = {np.max(np.abs(y)):.3f} (expect ~1.0, the 1 kHz tone)")


def test_zero_padding_effect():
    """Show that zero-padding refines the sampled response near the edges."""
    print("\n=== zero-padding effect on the 1500 Hz transition edge ===")
    # probe frequencies right around the upper cut
    probe = np.array([1450.0, 1480.0, 1500.0, 1520.0, 1550.0])
    coarse = fir.filter_freq_response(WINDOW, FS, BW, LOW_CUT, HIGH_CUT, probe, nfft=64)
    fine = fir.filter_freq_response(WINDOW, FS, BW, LOW_CUT, HIGH_CUT, probe, nfft=8192)
    print(f"{'freq Hz':>8} {'nfft=64':>9} {'nfft=8192':>10}")
    for f, c, fi in zip(probe, coarse, fine):
        print(f"{f:>8.0f} {c:>9.3f} {fi:>10.3f}")


def main() -> None:
    test_single_tones()
    test_multitone()
    test_zero_padding_effect()


if __name__ == "__main__":
    main()
