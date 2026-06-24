import os
import numpy as np
import raylib as rl
from scipy.signal import bilinear
import core.draw     as draw
import core.audio    as audio
import core.fir_core as fir

import core.scope   as scope

def render_frames_art(cfg, state):
    rl.ClearBackground((6, 12, 6))
    dt  = rl.GetFrameTime()
    scr = cfg.ScreenCfg

    render_scope_axis(scr)

    A, B              = 3,7           # Lissajous frequency ratio (sets the knobs)
    PHI_SPEED         = np.pi/2       # phase drift, rad/s (figure deforms)
    BEAM_SPEED        = 200.0         # parameter units traced per second
    SAMPLES_PER_FRAME = 4096          # beam resolution within one frame
    CAPACITY          = 4*4096        # trail length (persistence)

    buf = state.get("trail")
    if not isinstance(buf, draw.TrailBuffer):   # recreate on hot-reload / first run
        buf = draw.TrailBuffer(CAPACITY)
        state["trail"] = buf

    s0  = state.get("s",   0.0)
    phi = state.get("phi", 0.0)

    # advance the beam smoothly, sampling the arc covered this frame
    ds = BEAM_SPEED * dt
    s  = np.linspace(s0, s0 + ds, SAMPLES_PER_FRAME)
    x  = np.cos(A * s)
    y  = np.cos(B * s + phi)
    buf.extend(x, y)

    draw.draw_trail(scr, buf)

    state["s"]   = (s0 + ds) % (2*np.pi)   # A,B integer -> wraps with no jump
    state["phi"] = (phi + PHI_SPEED * dt) % (2*np.pi)


def render_scope_axis(cfg):
    N_DIVS = scope.Scope.n_divisions
    
    # Divisions Rendering
    for divX in range(N_DIVS):
        for divY in range(N_DIVS):
            XSect = divX / N_DIVS
            YSect = divY / N_DIVS
            XScaled = int(XSect * cfg.width())
            YScaled = int(YSect * cfg.height())
            rl.DrawRectangleLines(XScaled,YScaled,cfg.width()//N_DIVS,cfg.height()//N_DIVS,draw.SQ_GRAY)
    
    # Axis Markers Rendering
    N_MARKERS = scope.Scope.n_markers
    x_half = int(scope.Scope.mark_size_percent * (cfg.width() // N_DIVS) / 2)
    y_half = int(scope.Scope.mark_size_percent * (cfg.height() // N_DIVS) / 2)
    for mark in range(N_MARKERS):
        x_mark_center = (cfg.width()//2,int(mark/N_MARKERS * cfg.height()))
        rl.DrawLine(x_mark_center[0] - x_half, x_mark_center[1],
                    x_mark_center[0] + x_half, x_mark_center[1], draw.SQ_GRAY)

        y_mark_center = (int(mark/N_MARKERS * cfg.width()),cfg.height() // 2)
        rl.DrawLine(y_mark_center[0], y_mark_center[1] - y_half,
                    y_mark_center[0], y_mark_center[1] + y_half, draw.SQ_GRAY)

    # Axis 
    rl.DrawLine(0,cfg.height()//2,cfg.width(),cfg.height()//2,rl.WHITE)
    rl.DrawLine(cfg.width()//2,0,cfg.width()//2,cfg.height(),rl.WHITE)
    

def render_frames(cfg,state):
    rl.ClearBackground((12, 12, 12))
    dt = rl.GetFrameTime()
    
    speed = 0.25
    acc   = state.get("acc", 0.0) + dt
    period = 1.0 / speed

    BASE_F = 10
    df = state.get("df", 0)

    PHI_SPEED = np.pi/2               
    phi = (state.get("phi", 0.0) + PHI_SPEED * dt) % (2*np.pi)

    render_scope_axis(cfg.ScreenCfg)

    t = np.linspace(0.0, 1.0, scope.Scope.resolution)

    f = np.cos
    y = f(2*np.pi*BASE_F*t)
    z = f(2*np.pi*(BASE_F+df)*t + phi)
    
    show_xy = True
    scr = cfg.ScreenCfg

    if show_xy:
        draw.draw_polyline(scr, y, z, rl.PURPLE, state, "f_xy")
    else:
        draw.draw_polyline(scr, t, y, rl.YELLOW, state, "f_y")
        draw.draw_polyline(scr, t, z, rl.GREEN,  state, "f_z")

    while acc >= period:
        df = (df + 1) % (BASE_F)
        acc -= period

    state["acc"] = acc
    state["df"]  = df
    state["phi"] = phi

def render_frames_phase(cfg,state):
    rl.ClearBackground((12, 12, 12))
    dt = rl.GetFrameTime()
    
    speed = 20.0
    acc   = state.get("acc", 0.0) + dt
    period = 1.0 / speed

    PHI_STEP = np.pi/12          # phase advanced per transition
    phi = state.get("phi", 0.0)

    render_scope_axis(cfg.ScreenCfg)

    theta = np.linspace(-2*np.pi, 2*np.pi, scope.Scope.resolution)

    y = np.sin(theta)
    z = np.sin(theta + phi)
    
    show_xy = True
    scr = cfg.ScreenCfg

    if show_xy:
        draw.draw_polyline(scr, y, z, draw.SQ_PURPLE, state, "p_xy")
    else:
        draw.draw_polyline(scr, theta, y, rl.YELLOW, state, "p_y")
        draw.draw_polyline(scr, theta, z, rl.GREEN,  state, "p_z")

    while acc >= period:
        phi = (phi + PHI_STEP) % (2*np.pi)
        acc -= period

    state["acc"] = acc
    state["phi"] = phi


def render_frames_circles(cfg,state):
    rl.ClearBackground((12, 12, 12))
    
    render_scope_axis(cfg.ScreenCfg)
    
    CENTER1_ORIGIN = (-1.0,0)
    CENTER2_ORIGIN = (0.0,-1.0)
    
    CENTERS_SPEED = (1,5) # center1,center2 transitions/s

    center1 = state.get("center1",None) or CENTER1_ORIGIN
    center2 = state.get("center2",None) or CENTER2_ORIGIN

    if center1[0] > scope.Scope.x_max:
        center1 = CENTER1_ORIGIN

    if center2[1] > scope.Scope.y_max:
        center2 = CENTER2_ORIGIN

    trans_center1 = scope.Scope.translate(cfg.ScreenCfg,center1)
    trans_center2 = scope.Scope.translate(cfg.ScreenCfg,center2)

    rl.DrawCircle(trans_center1[0],trans_center1[1],10,rl.PINK)
    rl.DrawCircle(trans_center2[0],trans_center2[1],10,rl.BLUE)

    # discrete jumps: a "transition" is one update; CENTERS_SPEED is updates/s.
    dt = rl.GetFrameTime()
    STEP1 = 2 * scope.Scope.x_max / scope.Scope.n_markers
    STEP2 = 2 * scope.Scope.y_max / scope.Scope.n_markers

    acc1 = state.get("acc1", 0.0) + dt
    acc2 = state.get("acc2", 0.0) + dt
    period1 = 1.0 / CENTERS_SPEED[0]
    period2 = 1.0 / CENTERS_SPEED[1]

    while acc1 >= period1:
        center1 = (center1[0] + STEP1, center1[1])
        acc1 -= period1
    while acc2 >= period2:
        center2 = (center2[0], center2[1] + STEP2)
        acc2 -= period2

    state["acc1"] = acc1
    state["acc2"] = acc2
    state["center1"] = center1
    state["center2"] = center2


class Domain:
    '''
    The only things that differ between Laplace (S) and discrete (Z) bench:
      - how a real frequency maps to the complex evaluation point, and
      - which plane (s-plane axes vs z-plane unit circle) to draw.
    Everything else in the bench is shared.
    '''
    def __init__(self, kind, fs):
        self.kind = kind                 # "S" or "Z"
        self.fs   = float(fs)            # system sample rate (used only by Z)
        self.Ts   = 1.0 / float(fs)

    def pt_of_f(self, f):
        '''Evaluation point for a real frequency f: jw (S) or e^{jW} (Z).'''
        f = np.asarray(f, dtype=float)
        if self.kind == "Z":
            return np.exp(2j * np.pi * f * self.Ts)        # unit circle
        return 2j * np.pi * f                              # imaginary axis

    def probe_point(self, sigma, omega):
        '''Off-axis test point: s = sigma+jw (S) or z = e^{s*Ts} (Z).'''
        s = complex(sigma, omega)
        return np.exp(s * self.Ts) if self.kind == "Z" else s

    def draw_plane(self, *args):
        (draw.draw_zplane if self.kind == "Z" else draw.draw_splane)(*args)


def difference_equation_str(num, den, prec=6):
    '''
    Render a discrete TF  H(z) = num(z)/den(z)  (descending powers of z) as the
    recurrence that an implementation would actually run:

        a0*y[k] + a1*y[k-1] + ...  =  b0*x[k] + b1*x[k-1] + ...
        =>  y[k] = (1/a0)*( b0*x[k] + ... - a1*y[k-1] - ... )

    Coefficients are normalized by a0 so the leading y[k] term is implicit.
    '''
    b = np.atleast_1d(np.asarray(num, dtype=float))
    a = np.atleast_1d(np.asarray(den, dtype=float))
    b = b / a[0]
    a = a / a[0]

    terms = []                                   # (coeff, symbol), in display order
    for i, bi in enumerate(b):                   # feed-forward (input) taps
        if abs(bi) > 1e-12:
            terms.append((bi, f"x[k{f'-{i}' if i else ''}]"))
    for i, ai in enumerate(a[1:], start=1):      # feedback (output) taps, moved RHS => -ai
        if abs(ai) > 1e-12:
            terms.append((-ai, f"y[k-{i}]"))

    if not terms:
        return "y[k] = 0"

    out = "y[k] = "
    for j, (c, sym) in enumerate(terms):
        mag  = f"{abs(c):.{prec}g}*{sym}"
        if j == 0:
            out += f"-{mag}" if c < 0 else mag
        else:
            out += f" {'-' if c < 0 else '+'} {mag}"
    return out


def quantize(x, lo, hi, bits):
    '''
    Uniform mid-rise quantizer (an ideal ADC over a fixed voltage window).

    The interval [lo, hi] is the full-scale representation range and is split
    into  2**bits  equal levels of width  step = (hi-lo)/2**bits. Each sample is
    snapped to the center of its level and clamped to the range (saturation).
    Returns (xq, step) where step is the LSB (quantization voltage).
    '''
    levels = 1 << int(bits)
    step   = (hi - lo) / levels
    q      = np.floor((np.asarray(x, dtype=float) - lo) / step)
    q      = np.clip(q, 0, levels - 1)
    return lo + (q + 0.5) * step, step


def render_bench_tranfer_function(cfg, state,):
    '''
    Bench a Laplace transfer function H(s) = num(s)/den(s).

    `num`/`den` are polynomial coefficients in descending powers of s
    (scipy.signal convention). A "current" frequency is swept logarithmically
    from f_min to f_max; the window shows three panels:

        1. Amplitude response   |H(jw)|        in dB   (Bode, log-frequency)
        2. Phase response       /_H(jw)        in deg  (Bode, log-frequency)
        3. FFT of the signal currently under test (the filter's output at the
           swept frequency) -- its peak height tracks |H(jw)| as it sweeps.
    '''
    scr = cfg.ScreenCfg
    rl.ClearBackground((10, 10, 14))
    
    laplace_tfs = {
        "pass"     : ([1], [1]),
        "chebII_lp": (0.1*np.array([1,1.28,1e6]),[1,480,128000]),
        "chebI_hp" : ([1,0,0], [1.0, 4726, 1.666e7]),
        "chebI_bp" : ([7.3e7,0,0], 
                      np.polymul([1.0, 3341.9, 8.332e6],[1.0, 10134.9, 7.662e7])),
    }

    num, den = laplace_tfs["chebI_hp"]
    
    # num = den = None
    show_splane  = False
    show_fft     = False
    f_min        = 1.0
    f_max        = 8000
    sweep_period = 5.0
    quantization_bits = 8
    show_as_audio = False
    
    # Convert from s plane to z using bilinear projection => 
    # 2/T * (1 - z^-1) / (1 + z^-1)
    z_T          = 1/16000
    
    if num is None or den is None:
        wc  = 2 * np.pi * 1000.0
        num = [wc * wc]
        den = [1.0, wc*np.sqrt(2), wc*wc]
    
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    # ------------------------------------------------------------------
    # S vs Z domain. If z_T is given, discretize H(s) with the bilinear
    # transform  s = 2/T * (1 - z^-1)/(1 + z^-1)  (T == z_T == sample period).
    # The coefficients then live in z, and a real frequency f maps onto the
    # unit circle  z = e^{jwT}  instead of the imaginary axis  s = jw.
    # ------------------------------------------------------------------
    if z_T is not None:
        num, den = bilinear(num, den, fs=1.0 / z_T)      # -> discrete num(z)/den(z)
        s_to_pt  = lambda s: np.exp(np.asarray(s) * z_T)  # s-point (or jw) -> z
        eq_sig = num.tobytes() + den.tobytes()           # print recurrence once per change
        if state.get("_diffeq_sig") != eq_sig:
            state["_diffeq_sig"] = eq_sig
            print(f"[Z  T={z_T:g}s  fs={1/z_T:g}Hz]  {difference_equation_str(num, den)}")
    else:
        s_to_pt  = lambda s: np.asarray(s)                # identity: evaluate at s
    eval_pt = lambda w: s_to_pt(1j * np.asarray(w, dtype=float))

    # frequency response: constant for a fixed TF, so compute once and cache
    key  = (num.tobytes(), den.tobytes(), f_min, f_max, z_T)
    bode = state.get("bench_bode")
    if bode is None or bode[0] != key:
        freqs  = np.logspace(np.log10(f_min), np.log10(f_max), 1024)
        w      = 2 * np.pi * freqs
        H      = np.polyval(num, eval_pt(w)) / np.polyval(den, eval_pt(w))
        mag_db = 20 * np.log10(np.abs(H) + 1e-12)
        phase  = np.degrees(np.unwrap(np.angle(H)))
        bode   = (key, freqs, mag_db, phase)
        state["bench_bode"] = bode
    _, freqs, mag_db, phase = bode

    # advance the swept "current" frequency
    dt    = rl.GetFrameTime()
    u     = (state.get("bench_u", 0.0) + dt / sweep_period) % 1.0
    state["bench_u"] = u
    f_now = f_min * (f_max / f_min) ** u
    w_now = 2 * np.pi * f_now
    p_now = eval_pt(w_now)                       # jw (S) or e^{jwT} (Z)
    H_now = np.polyval(num, p_now) / np.polyval(den, p_now)

    # current readouts (phase interpolated from the unwrapped curve to match it)
    mag_now   = 20 * np.log10(np.abs(H_now) + 1e-12)
    phase_now = np.interp(f_now, freqs, phase)

    # play the displayed tone: sine at f_now scaled by the filter response
    # |H(f_now)|, with the same bit depth as the panel. Loud in passband, quiet
    # in stopband -- you literally hear the frequency response as it sweeps.
    if show_as_audio:
        snd = state.get("audio")
        if snd is None:
            snd = audio.ToneStreamer(sample_rate=44100, frames=2048)
            state["audio"] = snd
        snd.feed(f_now, amp=min(float(np.abs(H_now)), 1.0), bits=quantization_bits)
    else:
        snd = state.get("audio")
        if snd is not None:
            snd.stop()

    DECAY = 0.0                          
    M     = 2048
    fs    = 4.0 * f_max if not z_T else 1/(z_T)                  
    T     = M / fs                       
    t     = np.arange(M) / fs
    omega = w_now
    sigma = -DECAY / T                   # fixed real part (same at every frequency)
    s_now = complex(sigma, omega)
    pt_now = s_to_pt(s_now)              # s = sigma+jw (S)  or  z = e^{s T} (Z)
    
    # sig   = np.exp(sigma * t) * np.sin(omega * t)  * np.hamming(M)

    # H evaluated at the FULL complex point (general Laplace / z-plane point)
    H_s     = np.polyval(num, pt_now) / np.polyval(den, pt_now)
    mag_s   = 20 * np.log10(np.abs(H_s) + 1e-12)
    phase_s = np.degrees(np.angle(H_s))

    out = np.abs(H_now) * np.exp(sigma * t) * np.sin(omega * t + np.angle(H_now)) * np.hamming(M)

    # layout: three stacked panels
    margin, gap = 70, 45
    pw = scr.width() - 2 * margin
    ph = (scr.height() - 2 * margin - 2 * gap) // 3
    panels = [(margin, margin + i * (ph + gap), pw, ph) for i in range(3)]
    fr = (f_min, f_max)

    # 1+2) amplitude & phase  --  OR the equivalent pole-zero map in the s-plane
    
    if show_splane:
        r0, r1 = panels[0], panels[1]
        merged = (r0[0], r0[1], r0[2], (r1[1] + r1[3]) - r0[1])   # span both panels
        zeros  = np.roots(num) if num.size > 1 else np.array([])
        poles  = np.roots(den) if den.size > 1 else np.array([])
        if z_T is not None:
            title = (f"Plano Z   z = {pt_now.real:+.3f} {pt_now.imag:+.3f}j   "
                     f"|H(z)|={mag_s:+.1f} dB   /_H={phase_s:+.1f} deg")
            draw.draw_zplane(merged, poles, zeros, pt_now, title)
        else:
            title = (f"Plano S   s = {sigma:+.0f} {omega:+.0f}j   "
                     f"|H(s)|={mag_s:+.1f} dB   /_H={phase_s:+.1f} deg")
            draw.draw_splane(merged, poles, zeros, pt_now, title)
    else:
        # amplitude
        r  = panels[0]
        draw.draw_panel(r, f"Amplitude  |H(jw)|  [dB]      {mag_now:+7.1f} dB")
        yr = (float(mag_db.min()) - 5, float(mag_db.max()) + 5)
        draw.draw_log_grid(r, fr)
        draw.draw_h_grid(r, yr, 20.0)
        draw.vmarker(r, fr, f_now, draw.MARK_RED, xlog=True)
        px, py = draw.rect_map(freqs, mag_db, fr, yr, r, xlog=True)
        draw.draw_curve(state, "bench_mag", px, py, draw.SQ_PURPLE, rect=r)

        # phase
        r  = panels[1]
        draw.draw_panel(r, f"Fase  /_H(jw)  [graus]      {phase_now:+7.1f} deg")
        yr = (float(phase.min()) - 10, float(phase.max()) + 10)
        draw.draw_log_grid(r, fr)
        draw.draw_h_grid(r, yr, 45.0)
        draw.vmarker(r, fr, f_now, draw.MARK_RED, xlog=True)
        px, py = draw.rect_map(freqs, phase, fr, yr, r, xlog=True)
        draw.draw_curve(state, "bench_phase", px, py, (120, 220, 160, 255), rect=r)

    # 3) the tested signal in time  --  OR its FFT
    r  = panels[2]
    sig_plot = out
    
    if show_fft:
        # quantize before the transform so the spectrum shows the quantization
        # noise floor (SNR ~ 6.02*bits + 1.76 dB). Same full-scale window as the
        # time plot below: [-1.1, 1.1] V.
        q_title = ""
        if quantization_bits is not None:
            sig_plot, lsb = quantize(out, -1.1, 1.1, quantization_bits)
            q_title = f"  [{int(quantization_bits)} bits  LSB={lsb:.4g} V]"

        win    = sig_plot * np.hanning(M)
        spec   = np.abs(np.fft.rfft(win)) / (M / 2)
        spec_f = np.fft.rfftfreq(M, 1 / fs)
        draw.draw_panel(r, f"FFT do sinal testado   (f = {f_now:8.1f} Hz)" + q_title)
        xr = (0.0, f_max/4)
        yr = (0.0, 1.1)
        draw.draw_h_grid(r, yr, yr[1] / 4)
        draw.vmarker(r, xr, f_now, (255, 90, 90, 120), xlog=False)
        px, py = draw.rect_map(spec_f, spec, xr, yr, r, xlog=False)
        draw.draw_curve(state, "bench_fft", px, py, (255, 210, 80, 255), rect=r)
    else:
        xr = (0, T)
        yr = (-1.1, 1.1)                      # full-scale voltage window for the ADC

        if quantization_bits is not None:
            sig_plot, lsb = quantize(out, yr[0], yr[1], quantization_bits)
            draw.draw_panel(r, f"Sinal quantizado   {int(quantization_bits)} bits   "
                               f"{1 << int(quantization_bits)} niveis   LSB={lsb:.4g} V   "
                               f"f={f_now:7.1f} Hz")
            draw.draw_h_grid(r, yr, lsb)
        else:
            draw.draw_panel(r, f"Sinal testado  e^(sigma t) sin(w t)   "
                               f"sigma={sigma:+.0f}/s   f={f_now:7.1f} Hz")
            draw.draw_h_grid(r, yr, 0.5)

        px, py = draw.rect_map(t, sig_plot, xr, yr, r, xlog=False)
        draw.draw_curve(state, "bench_signal", px, py, (255, 210, 80, 255), rect=r,
                        stair=(z_T is not None or quantization_bits is not None))

def render_fir_bench(cfg, state):
    '''
    Bench a windowed-sinc FIR filter designed with core.fir_core.

    The filter is specified by a window, a pass band [low_cut, high_cut] and a
    transition bandwidth `bw` (which sets the tap count N via the window's rule).
    Three panels, mirroring render_bench_tranfer_function:

        1. Amplitude |H(f)|  [dB]            (or the impulse response h[n] when
        2. Phase     /_H(f)  [graus]          show_taps is on -- panels 1+2 merge)
        3. A pure tone at the swept frequency, ACTUALLY filtered through the FIR
           (spectrum * H, exactly like fir_freq_response.apply_filter) -- shown
           in time, or as its FFT.
    '''
    scr = cfg.ScreenCfg
    rl.ClearBackground((10, 10, 14))

    # BLOCKS SIZES TO CONSIDER THE FFT TAIL
    # B    = nfft - (N - 1)            # usable block = 892 samples per FFT

    # ---- FIR specification -------------------------------------------------
    fs        = 16000.0         # filter sample rate
    low_cut   = 0.0             # pass band lower edge (0 -> low-pass)
    high_cut  = 800.0           # pass band upper edge (>= fs/2 -> high-pass)
    bw        = 100.0           # transition bandwidth -> tap count N
    window    = fir.Blackman    # Hamming / Hanning / Blackman / Rectangular
    N_FFT     = 2048

    show_taps    = False        # merge panels 0+1 and draw h[n] instead of mag/phase
    show_fft     = False        # panel 3: FFT instead of the time-domain output
    f_min        = 20.0
    f_max        = fs / 2.0     # sweep up to Nyquist
    sweep_period = 10
    quantization_bits = 8
    show_as_audio     = True

    # ---- design (cached): taps + exact complex response over the sweep -----
    key    = (window.__name__, fs, low_cut, high_cut, bw, f_min, f_max)
    design = state.get("fir_design")
    if design is None or design[0] != key:
        h, N, M = fir.fir_taps(window, fs, bw, low_cut, high_cut,N_FFT)
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), 1024)
        # exact FIR response  H(f) = sum_n h[n] e^{-j 2*pi*f*n/fs}
        E      = np.exp(-2j * np.pi * np.outer(freqs, np.arange(h.size)) / fs)
        Hc     = E @ h
        mag_db = 20 * np.log10(np.abs(Hc) + 1e-12)
        phase  = np.degrees(np.unwrap(np.angle(Hc)))
        design = (key, h, N, M, freqs, mag_db, phase)
        state["fir_design"] = design
        print(f"[FIR {window.__name__}  N={N} taps  atraso de grupo={M} amostras "
              f"({M / fs * 1e3:.2f} ms)  passa {low_cut:g}..{high_cut:g} Hz @ {fs:g} Hz]")
    _, h, N, M, freqs, mag_db, phase = design

    # press E -> export the CURRENT design to filtro_fir_esp32/fir_taps.h, which
    # filtro_fir_esp32.ino #includes at compile time (overlap-save FIR on ESP32).
    if rl.IsKeyPressed(rl.KEY_E):
        proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dest = os.path.join(proj, "filtro_fir_esp32", "fir_taps.h")
        fir.write_taps_header(dest, h, N, M, N_FFT, fs,
                              window=window, low_cut=low_cut, high_cut=high_cut, bw=bw)
        print(f"[exportado] {dest}  (N={N} taps, N_FFT={N_FFT}, B={N_FFT-(N-1)})")

    # ---- swept "current" frequency ----------------------------------------
    dt    = rl.GetFrameTime()
    u     = (state.get("fir_u", 0.0) + dt / sweep_period) % 1.0
    state["fir_u"] = u
    f_now = f_min * (f_max / f_min) ** u

    H_now     = np.sum(h * np.exp(-2j * np.pi * f_now * np.arange(h.size) / fs))  # exact
    mag_now   = 20 * np.log10(np.abs(H_now) + 1e-12)
    phase_now = np.interp(f_now, freqs, phase)

    # audio: play f_now scaled by |H(f_now)| -> loud in band, quiet out of band
    if show_as_audio:
        snd = state.get("audio")
        if snd is None:
            snd = audio.ToneStreamer(sample_rate=44100, frames=2048)
            state["audio"] = snd
        snd.feed(f_now, amp=min(float(np.abs(H_now)), 1.0), bits=quantization_bits)
    else:
        snd = state.get("audio")
        if snd is not None:
            snd.stop()

    # ---- actually FILTER a test tone through the FIR (uses fir_core) -------
    # mirror fir_freq_response.apply_filter: spectrum * |H| then inverse FFT.
    Msig      = 2048
    t         = np.arange(Msig) / fs
    x         = np.cos(2 * np.pi * f_now * t) * np.hamming(Msig) # pure tone at f_now
    X         = np.fft.rfft(x)
    sig_freqs = np.fft.rfftfreq(Msig, d=1.0 / fs)
    Hmag      = fir.filter_freq_response(window, fs, bw, low_cut, high_cut, sig_freqs)
    y         = np.fft.irfft(X * Hmag, n=Msig)               # zero-phase filtered output

    # ---- layout: three stacked panels -------------------------------------
    margin, gap = 70, 45
    pw = scr.width() - 2 * margin
    ph = (scr.height() - 2 * margin - 2 * gap) // 3
    panels = [(margin, margin + i * (ph + gap), pw, ph) for i in range(3)]
    fr = (f_min, f_max)

    # ---- panels 0/1: amplitude + phase  OR  the impulse response ----------
    if show_taps:
        r0, r1 = panels[0], panels[1]
        merged = (r0[0], r0[1], r0[2], (r1[1] + r1[3]) - r0[1])   # span both panels
        draw.draw_panel(merged, f"Resposta ao impulso  h[n]   {N} taps   "
                                f"{window.__name__}   atraso de grupo {M} amostras")
        taps = h[:N]                                   # only the real taps (drop padding)
        hr = max(float(np.abs(taps).max()), 1e-6) * 1.2
        xr = (0.0, N - 1);  yr = (-hr, hr)
        draw.draw_h_grid(merged, yr, draw._nice_step(2 * hr))
        draw.vmarker(merged, xr, M, (255, 90, 90, 120), xlog=False)   # center tap
        px, py = draw.rect_map(np.arange(N), taps, xr, yr, merged, xlog=False)
        draw.draw_curve(state, "fir_taps", px, py, draw.SQ_PURPLE, rect=merged, stair=True)
    else:
        # amplitude
        r  = panels[0]
        draw.draw_panel(r, f"Amplitude  |H(f)|  [dB]      {mag_now:+7.1f} dB   ({N} taps)")
        yr = (max(float(mag_db.min()), -120.0) - 5, float(mag_db.max()) + 5)
        draw.draw_log_grid(r, fr)
        draw.draw_h_grid(r, yr, 20.0)
        for fc in (low_cut, high_cut):
            if f_min < fc < f_max:
                draw.vmarker(r, fr, fc, (90, 160, 255, 120), xlog=True)   # cutoffs
        draw.vmarker(r, fr, f_now, draw.MARK_RED, xlog=True)
        px, py = draw.rect_map(freqs, mag_db, fr, yr, r, xlog=True)
        draw.draw_curve(state, "fir_mag", px, py, draw.SQ_PURPLE, rect=r)

        # phase (unwrapped -> a near-straight ramp == the linear-phase signature)
        r  = panels[1]
        draw.draw_panel(r, f"Fase  /_H(f)  [graus]      {phase_now:+8.1f} deg")
        yr = (float(phase.min()) - 10, float(phase.max()) + 10)
        draw.draw_log_grid(r, fr)
        draw.draw_h_grid(r, yr, draw._nice_step(yr[1] - yr[0]))
        draw.vmarker(r, fr, f_now, draw.MARK_RED, xlog=True)
        px, py = draw.rect_map(freqs, phase, fr, yr, r, xlog=True)
        draw.draw_curve(state, "fir_phase", px, py, (120, 220, 160, 255), rect=r)

    # ---- panel 2: the filtered signal in time  --  OR its FFT -------------
    r        = panels[2]
    sig_plot = y
    if show_fft:
        q_title = ""
        if quantization_bits is not None:
            sig_plot, lsb = quantize(y, -1.1, 1.1, quantization_bits)
            q_title = f"  [{int(quantization_bits)} bits  LSB={lsb:.4g} V]"
        win    = sig_plot * np.hanning(Msig)
        spec   = np.abs(np.fft.rfft(win)) / (Msig / 2)
        spec_f = np.fft.rfftfreq(Msig, 1 / fs)
        draw.draw_panel(r, f"FFT do sinal filtrado   (f = {f_now:8.1f} Hz)" + q_title)
        xr = (0.0, f_max);  yr = (0.0, 1.1)
        draw.draw_h_grid(r, yr, yr[1] / 4)
        for fc in (low_cut, high_cut):
            draw.vmarker(r, xr, fc, (90, 160, 255, 120), xlog=False)
        draw.vmarker(r, xr, f_now, (255, 90, 90, 120), xlog=False)
        px, py = draw.rect_map(spec_f, spec, xr, yr, r, xlog=False)
        draw.draw_curve(state, "fir_fft", px, py, (255, 210, 80, 255), rect=r)
    else:
        xr = (0.0, Msig / fs);  yr = (-1.1, 1.1)
        if quantization_bits is not None:
            sig_plot, lsb = quantize(y, yr[0], yr[1], quantization_bits)
            draw.draw_panel(r, f"Sinal filtrado quantizado   {int(quantization_bits)} bits   "
                               f"LSB={lsb:.4g} V   f={f_now:7.1f} Hz")
            draw.draw_h_grid(r, yr, lsb)
        else:
            draw.draw_panel(r, f"Sinal filtrado pela FIR   y[n]   f={f_now:7.1f} Hz   "
                               f"ganho |H|={float(np.abs(H_now)):.3f}")
            draw.draw_h_grid(r, yr, 0.5)
        px, py = draw.rect_map(t, sig_plot, xr, yr, r, xlog=False)
        draw.draw_curve(state, "fir_signal", px, py, (255, 210, 80, 255), rect=r,
                        stair=(quantization_bits is not None))


# Active renderer (assigned after all renderers are defined).
render_target = render_bench_tranfer_function
# render_target = render_bench_tranfer_function
