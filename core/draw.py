from core.configs import ScreenCfg

import core.scope as scope           # module-qualified so reload(scope) propagates here

import numpy as np
import raylib as rl

SQ_GRAY    = (130, 130, 130, 127)
SQ_PURPLE  = (182, 151, 255, 255)
PANEL_BG   = (12 ,  12,  18, 255)
GRID_MAJOR = (60 ,  60,  70, 255)
GRID_MINOR = (32 ,  32,  40, 255)
TXT_DIM    = (120, 120, 135, 255)
MARK_RED   = (255,  90,  90, 200)


class TrailBuffer:
    '''
    Ring buffer of beam positions (in scope coords) for phosphor-style fade.
    Newest samples are drawn bright, oldest dim, then overwritten.
    '''
    def __init__(self, capacity:int):
        self.capacity = capacity
        self.xy    = np.zeros((capacity, 2), dtype=np.float64)
        self.head  = 0      # index of the next write slot
        self.count = 0      # number of valid samples so far

    def extend(self, xs, ys):
        '''Append a batch of samples, overwriting the oldest when full.'''
        pts = np.column_stack((xs, ys))
        n   = len(pts)
        if n >= self.capacity:                       #render_target = render_frames_art batch alone overflows: keep tail
            self.xy[:]  = pts[-self.capacity:]
            self.head   = 0
            self.count  = self.capacity
            return
        end = self.head + n
        if end <= self.capacity:                     # contiguous write
            self.xy[self.head:end] = pts
        else:                                        # wraps past the end
            first = self.capacity - self.head
            self.xy[self.head:] = pts[:first]
            self.xy[:n - first] = pts[first:]
        self.head  = end % self.capacity
        self.count = min(self.count + n, self.capacity)

    def ordered(self):
        '''Samples from oldest to newest.'''
        if self.count < self.capacity:
            return self.xy[:self.count]
        return np.roll(self.xy, -self.head, axis=0)


def draw_trail(cfg:ScreenCfg, buf:TrailBuffer, base_color=(255,255,20), tau=0.35):
    '''
    Draw the ring buffer as a fading polyline: brightness decays toward the
    oldest sample (exponential, controlled by tau in [0,1]).
    '''
    pts = buf.ordered()
    n   = len(pts)
    if n < 2:
        return

    flipped = pts * np.array((1, -1))
    scaled  = (0.5 + flipped / (2 * np.array((scope.Scope.x_max, scope.Scope.y_max)))) \
              * np.array((cfg.width(), cfg.height()))
    screen  = scaled.astype(int)

    r, g, b = base_color
    for i in range(n - 1):
        frac  = (i + 1) / (n - 1)                    # 0 = oldest, 1 = newest
        alpha = int(255 * np.exp(-(1.0 - frac) / tau))
        rl.DrawLine(screen[i][0],   screen[i][1],
                    screen[i+1][0], screen[i+1][1], (r, g, b, alpha))


def strip_buffer(state, key, n):
    '''
    Cache a reusable cffi Vector2[] (and a float32 numpy view over its memory)
    so we don't reallocate every frame. Rebuilt only when the point count changes.
    '''
    cache = state.get(key)
    if cache is None or cache[0] != n:
        pts  = rl.ffi.new("Vector2[]", n)
        view = np.frombuffer(rl.ffi.buffer(pts), dtype=np.float32)  # interleaved x,y
        cache = (n, pts, view)
        state[key] = cache
    return cache[1], cache[2]


def draw_polyline(cfg, xs, ys, color, state, key, z_period = None):
    '''
    Draw a connected polyline from scope coords in ONE C call.

    The whole coordinate transform is vectorized over the arrays and written
    straight into the raylib vertex buffer, so per-frame Python work is O(1)
    regardless of resolution.
    '''
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n  = len(xs)
    if n < 2:
        return

    # Zero-order hold (sample-and-hold) trace for the discrete/z domain: every
    # segment (x1,y1)->(x2,y2) becomes (x1,y1)->(x2,y1)->(x2,y2), so the value is
    # held flat across one sample period (z_period == z_T) and only steps at the
    # next sample. Doubles the vertex count to 2n-1.
    if z_period is not None:
        m  = 2 * n - 1
        sx = np.empty(m); sy = np.empty(m)
        sx[0], sy[0] = xs[0], ys[0]
        sx[1::2] = xs[1:]; sx[2::2] = xs[1:]   # hold x, then step x
        sy[1::2] = ys[:-1]; sy[2::2] = ys[1:]  # hold y, then step y
        xs, ys, n = sx, sy, m

    pts, view = strip_buffer(state, key, n)
    view[0::2] = (0.5 + xs / (2 * scope.Scope.x_max)) * cfg.width()
    view[1::2] = (0.5 - ys / (2 * scope.Scope.y_max)) * cfg.height()  # y is flipped
    rl.DrawLineStrip(pts, n, color)


def rect_map(x, y, xr, yr, rect, xlog=False):
    '''Map data (x,y) into pixel coords inside `rect`=(rx,ry,rw,rh).'''
    rx, ry, rw, rh = rect
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if xlog:
        xmin, xmax = np.log10(xr[0]), np.log10(xr[1])
        xs = np.log10(np.maximum(x, 1e-12))
    else:
        xmin, xmax = xr
        xs = x
    ymin, ymax = yr
    px = rx + (xs - xmin) / (xmax - xmin) * rw
    py = ry + rh - (y - ymin) / (ymax - ymin) * rh        # invert: ymin at bottom
    return px, py


def draw_curve(state, key, px, py, color, rect=None, stair=False):
    '''
    Draw a polyline (one C call), optionally clipped to `rect`.

    With `stair=True` the samples are drawn zero-order-hold (sample-and-hold):
    each (x1,y1)->(x2,y2) becomes (x1,y1)->(x2,y1)->(x2,y2), so the value is held
    flat until the next sample steps. Use this for discrete/z-domain sequences.
    '''
    px = np.asarray(px)
    py = np.asarray(py)
    n  = len(px)
    if n < 2:
        return
    if stair:
        m  = 2 * n - 1
        sx = np.empty(m); sy = np.empty(m)
        sx[0], sy[0] = px[0], py[0]
        sx[1::2] = px[1:]; sx[2::2] = px[1:]   # hold x, then step x
        sy[1::2] = py[:-1]; sy[2::2] = py[1:]  # hold y, then step y
        px, py, n = sx, sy, m
    pts, view = strip_buffer(state, key, n)
    view[0::2] = px
    view[1::2] = py
    if rect is not None:
        rl.BeginScissorMode(int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))
    rl.DrawLineStrip(pts, n, color)
    if rect is not None:
        rl.EndScissorMode()

def draw_panel(rect, title):
    rx, ry, rw, rh = rect
    rl.DrawRectangle(rx, ry, rw, rh, PANEL_BG)
    rl.DrawRectangleLines(rx, ry, rw, rh, SQ_GRAY)
    rl.DrawText(title.encode("utf-8"), rx + 10, ry - 20, 20, (200, 200, 210, 255))


def draw_log_grid(rect, xr):
    '''Decade/minor vertical grid for a log-frequency axis, with decade labels.'''
    rx, ry, rw, rh = rect
    xmin, xmax = np.log10(xr[0]), np.log10(xr[1])
    for d in range(int(np.floor(xmin)), int(np.ceil(xmax)) + 1):
        for m in range(1, 10):
            f = m * 10.0**d
            if f < xr[0] or f > xr[1]:
                continue
            x = int(rx + (np.log10(f) - xmin) / (xmax - xmin) * rw)
            rl.DrawLine(x, ry, x, ry + rh, GRID_MAJOR if m == 1 else GRID_MINOR)
            if m == 1:
                rl.DrawText(f"{f:g}".encode(), x + 3, ry + rh - 16, 14, TXT_DIM)


def draw_h_grid(rect, yr, step):
    '''Horizontal grid + value labels every `step` units of y.'''
    rx, ry, rw, rh = rect
    ymin, ymax = yr
    v = np.ceil(ymin / step) * step
    while v <= ymax:
        y = int(ry + rh - (v - ymin) / (ymax - ymin) * rh)
        rl.DrawLine(rx, y, rx + rw, y, GRID_MINOR)
        rl.DrawText(f"{v:g}".encode(), rx + 4, y - 16, 14, TXT_DIM)
        v += step


def vmarker(rect, xr, f, color, xlog=True):
    rx, ry, rw, rh = rect
    if xlog:
        xmin, xmax = np.log10(xr[0]), np.log10(xr[1])
        frac = (np.log10(f) - xmin) / (xmax - xmin)
    else:
        frac = (f - xr[0]) / (xr[1] - xr[0])
    x = int(rx + frac * rw)
    rl.DrawLine(x, ry, x, ry + rh, color)


def _draw_x(cx, cy, r, color):
    rl.DrawLine(cx - r, cy - r, cx + r, cy + r, color)
    rl.DrawLine(cx - r, cy + r, cx + r, cy - r, color)


def _nice_step(span, target=8):
    '''Round span/target to the nearest 1/2/5 x 10^k for clean grid spacing.'''
    raw = span / target
    p   = 10.0 ** np.floor(np.log10(raw))
    return min((1, 2, 5, 10), key=lambda m: abs(m - raw / p)) * p


SPLANE_POLE = (255, 90, 90, 255)
SPLANE_ZERO = (120, 200, 255, 255)
SPLANE_TEST = (255, 210, 80, 255)


def draw_splane(rect, poles, zeros, s_now, title):
    '''
    Pole-zero map of H(s) in the complex s-plane (sigma horizontal, jw vertical).

    The test point s = sigma + j*omega moves in BOTH axes: sigma (real) is the
    envelope growth/decay rate of the probe, omega (imaginary) its frequency.
    Faint guide lines drop from the point to each axis so the real/imaginary
    decomposition is visible, and vectors from every pole/zero to the point show
    the geometry behind |H| (product of distances) and phase (sum of angles).
    '''
    rx, ry, rw, rh = rect
    draw_panel(rect, title)

    poles = np.asarray(poles, dtype=complex)
    zeros = np.asarray(zeros, dtype=complex)

    # symmetric range fixed by the poles/zeros only (NOT the test point), so the
    # pole-zero map stays put while s sweeps off the edges.
    feats = np.concatenate((poles, zeros))
    rng   = max(float(np.abs(feats).max()) * 1.2, 1.0) if feats.size else 1.0

    asp = rw / rh                                  # keep 1 unit == 1 unit (equal aspect)
    if asp >= 1:
        yr, xr = (-rng, rng), (-rng * asp, rng * asp)
    else:
        xr, yr = (-rng, rng), (-rng / asp, rng / asp)

    rl.BeginScissorMode(rx, ry, rw, rh)

    # horizontal grid (lines of constant jw), behind everything
    draw_h_grid(rect, yr, _nice_step(yr[1] - yr[0]))

    # axes through the origin
    (ox,), (oy,) = rect_map([0.0], [0.0], xr, yr, rect)
    ox, oy = int(ox), int(oy)
    rl.DrawLine(rx, oy, rx + rw, oy, GRID_MAJOR)   # real (sigma) axis
    rl.DrawLine(ox, ry, ox, ry + rh, GRID_MAJOR)   # imaginary (jw) axis
    rl.DrawText(b"jw", ox + 6, ry + 4, 14, TXT_DIM)
    rl.DrawText(b"sigma", rx + rw - 48, oy + 6, 14, TXT_DIM)

    # current test point s = sigma + j*omega (anywhere in the plane)
    (sx,), (sy,) = rect_map([s_now.real], [s_now.imag], xr, yr, rect)
    sx, sy = int(sx), int(sy)

    # guide lines: drop to the imaginary axis (-> reads sigma) and to the real
    # axis (-> reads omega), making the real/imaginary split explicit
    rl.DrawLine(sx, sy, ox, sy, (255, 210, 80, 90))   # horizontal: real part sigma
    rl.DrawLine(sx, sy, sx, oy, (255, 210, 80, 90))   # vertical:   imag part omega
    rl.DrawCircle(ox, sy, 4, (255, 120, 120, 200))    # sigma read-off on jw axis
    rl.DrawCircle(sx, oy, 4, (120, 200, 255, 200))    # omega read-off on sigma axis

    # geometric vectors from each pole/zero to the test point
    if poles.size:
        ppx, ppy = rect_map(poles.real, poles.imag, xr, yr, rect)
        for x, y in zip(ppx, ppy):
            rl.DrawLine(int(x), int(y), sx, sy, (255, 90, 90, 70))
    if zeros.size:
        zpx, zpy = rect_map(zeros.real, zeros.imag, xr, yr, rect)
        for x, y in zip(zpx, zpy):
            rl.DrawLine(int(x), int(y), sx, sy, (120, 200, 255, 70))

    # poles as X, zeros as O
    if poles.size:
        for x, y in zip(ppx, ppy):
            _draw_x(int(x), int(y), 9, SPLANE_POLE)
    if zeros.size:
        for x, y in zip(zpx, zpy):
            rl.DrawCircleLines(int(x), int(y), 9, SPLANE_ZERO)

    rl.DrawCircle(sx, sy, 6, SPLANE_TEST)
    rl.EndScissorMode()


def draw_zplane(rect, poles, zeros, z_now, title):
    '''
    Pole-zero map of H(z) in the complex z-plane. Same geometry idea as
    draw_splane, but the frequency axis AND the stability boundary are the UNIT
    CIRCLE |z| = 1: the test point z = r e^{jW} rides the circle, and poles inside
    the circle are stable. |H| and phase are still products/sums of the pole/zero
    vectors to the test point, drawn faintly.
    '''
    rx, ry, rw, rh = rect
    draw_panel(rect, title)

    poles = np.asarray(poles, dtype=complex)
    zeros = np.asarray(zeros, dtype=complex)

    # range fixed by poles/zeros AND the unit circle, so the map stays put
    feats = np.concatenate((poles, zeros, [1 + 0j]))
    rng   = max(float(np.abs(feats).max()) * 1.15, 1.15)

    asp = rw / rh                                  # equal aspect -> circle looks round
    if asp >= 1:
        yr, xr = (-rng, rng), (-rng * asp, rng * asp)
    else:
        xr, yr = (-rng, rng), (-rng / asp, rng / asp)

    rl.BeginScissorMode(rx, ry, rw, rh)
    draw_h_grid(rect, yr, _nice_step(yr[1] - yr[0]))

    # axes through the origin
    (ox,), (oy,) = rect_map([0.0], [0.0], xr, yr, rect)
    ox, oy = int(ox), int(oy)
    rl.DrawLine(rx, oy, rx + rw, oy, GRID_MAJOR)   # real axis
    rl.DrawLine(ox, ry, ox, ry + rh, GRID_MAJOR)   # imaginary axis
    rl.DrawText(b"Im", ox + 6, ry + 4, 14, TXT_DIM)
    rl.DrawText(b"Re", rx + rw - 24, oy + 6, 14, TXT_DIM)

    # unit circle: digital-frequency axis & stability boundary
    r_px = rw / (xr[1] - xr[0])                     # pixels per unit (equal aspect)
    rl.DrawCircleLines(ox, oy, r_px, GRID_MAJOR)

    # current test point z = r e^{jW}
    (sx,), (sy,) = rect_map([z_now.real], [z_now.imag], xr, yr, rect)
    sx, sy = int(sx), int(sy)

    # geometric vectors from each pole/zero to the test point
    if poles.size:
        ppx, ppy = rect_map(poles.real, poles.imag, xr, yr, rect)
        for x, y in zip(ppx, ppy):
            rl.DrawLine(int(x), int(y), sx, sy, (255, 90, 90, 70))
    if zeros.size:
        zpx, zpy = rect_map(zeros.real, zeros.imag, xr, yr, rect)
        for x, y in zip(zpx, zpy):
            rl.DrawLine(int(x), int(y), sx, sy, (120, 200, 255, 70))

    # poles as X, zeros as O
    if poles.size:
        for x, y in zip(ppx, ppy):
            _draw_x(int(x), int(y), 9, SPLANE_POLE)
    if zeros.size:
        for x, y in zip(zpx, zpy):
            rl.DrawCircleLines(int(x), int(y), 9, SPLANE_ZERO)

    rl.DrawCircle(sx, sy, 6, SPLANE_TEST)
    rl.EndScissorMode()