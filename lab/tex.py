"""Texture math: tileable noise, wear, grime, normals from height, image IO. Pure numpy/PIL (no bpy).
Arrays are float32 in 0..1, shape (H, W) or (H, W, C). Row 0 = top of the image = UV v=1."""
import os
import numpy as np
from PIL import Image

RNG = np.random.default_rng


def save(arr, path, mode=None, bits=8):
    a = np.clip(np.asarray(arr, np.float32), 0, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if bits == 16 and a.ndim == 2:
        Image.fromarray((a * 65535 + 0.5).astype(np.uint16)).save(path)
    else:
        Image.fromarray((a * 255 + 0.5).astype(np.uint8), mode).save(path)
    return path


def load(path):
    return np.asarray(Image.open(path), np.float32) / 255.0


def srgb_to_lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, np.float32), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


# ---------------------------------------------------------------- noise (tileable, spectral)
def spectral(h, w, fmin=1.0, fmax=64.0, beta=2.0, seed=0, ax=1.0):
    """Tileable 1/f^beta noise in [0,1]. f in cycles per image. ax<1 elongates features along x."""
    r = RNG(seed)
    fy = np.fft.fftfreq(h)[:, None] * h
    fx = np.fft.rfftfreq(w)[None, :] * w
    f = np.sqrt((fx * ax) ** 2 + fy ** 2)
    amp = np.where((f >= fmin) & (f <= fmax), 1.0 / np.maximum(f, 1e-6) ** (beta / 2), 0.0)
    ph = r.uniform(0, 2 * np.pi, amp.shape)
    n = np.fft.irfft2(amp * np.exp(1j * ph), s=(h, w))
    n -= n.min()
    return (n / max(n.max(), 1e-9)).astype(np.float32)


def fbm(h, w, scale=8, seed=0, beta=2.0, ax=1.0):
    return spectral(h, w, fmin=max(scale / 4, 1), fmax=scale * 16, beta=beta, seed=seed, ax=ax)


def remap(a, lo, hi):
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def blur(a, px):
    """Separable box-ish blur via FFT gaussian, wraps (tileable)."""
    if px <= 0:
        return a
    h, w = a.shape[:2]
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    g = np.exp(-2 * (np.pi * px) ** 2 * (fx ** 2 + fy ** 2))
    if a.ndim == 2:
        return np.real(np.fft.ifft2(np.fft.fft2(a) * g)).astype(np.float32)
    return np.stack([blur(a[..., i], px) for i in range(a.shape[2])], -1)


# ---------------------------------------------------------------- surface detail
def scratches(h, w, n=200, length=(0.01, 0.08), width=1, seed=0, angle=None):
    """Thin random scratches as a 0..1 mask (anti-aliased by supersampling)."""
    from PIL import ImageDraw
    r = RNG(seed)
    ss = 2
    im = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(im)
    for _ in range(n):
        x, y = r.uniform(0, w * ss), r.uniform(0, h * ss)
        a = r.uniform(0, np.pi) if angle is None else angle + r.normal(0, 0.25)
        L = r.uniform(*length) * w * ss
        v = int(r.uniform(90, 255))
        d.line([x, y, x + np.cos(a) * L, y + np.sin(a) * L], fill=v, width=max(1, int(width * ss * r.uniform(0.6, 1.3))))
    return np.asarray(im.resize((w, h), Image.LANCZOS), np.float32) / 255.0


def speckle(h, w, density=0.002, seed=0, size=1.5):
    r = RNG(seed)
    m = (r.random((h, w)) < density).astype(np.float32)
    return np.clip(blur(m, size) * size * size * 6, 0, 1)


def height_to_normal(hgt, strength=1.0):
    """OpenGL tangent-space normal map (green up) from a height map; tileable gradients."""
    dx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * 0.5 * strength
    dy = (np.roll(hgt, 1, 0) - np.roll(hgt, -1, 0)) * 0.5 * strength   # row 0 is v=1
    n = np.stack([-dx, -dy, np.ones_like(hgt)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def blend_normals(base, detail):
    """Reoriented normal mapping, both in 0..1 encoding."""
    t = base * np.array([2, 2, 2]) + np.array([-1, -1, 0])
    u = detail * np.array([-2, -2, 2]) + np.array([1, 1, -1])
    r = t * np.sum(t * u, -1, keepdims=True) / t[..., 2:3] - u
    r /= np.linalg.norm(r, axis=-1, keepdims=True)
    return r * 0.5 + 0.5


def edge_wear(curv, ao, seed=0, amount=0.5, scale=24):
    """Wear mask: convex edges (curvature) broken up by noise, plus rub on exposed areas. All 0..1."""
    h, w = curv.shape
    n = fbm(h, w, scale, seed)
    chips = remap(curv * (0.6 + 0.8 * n) - (1 - amount) * 0.7, 0.0, 0.25)
    rub = remap(ao * fbm(h, w, scale / 3, seed + 1) - 0.75, 0, 0.25) * 0.3 * amount
    return np.clip(chips + rub, 0, 1)


def dist_to_poly(x, y, poly):
    """Unsigned distance from points (arrays x, y) to a closed polygon's edges (same units)."""
    poly = np.asarray(poly, np.float32)
    d = np.full(x.shape, np.inf, np.float32)
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1]) / max(float(ab @ ab), 1e-12), 0, 1)
        d = np.minimum(d, np.hypot(x - (a[0] + t * ab[0]), y - (a[1] + t * ab[1])))
    return d


# ---------------------------------------------------------------- UV-space helpers (lathe islands)
def uv_grid(S):
    j, i = np.meshgrid(np.arange(S), np.arange(S))
    return ((j + 0.5) / S).astype(np.float32), (1 - (i + 0.5) / S).astype(np.float32)


def in_rect(u, v, r):
    return (u >= r[0]) & (u < r[2]) & (v >= r[1]) & (v < r[3])


def side_map(u, v, rect, prof):
    """For a lathe 'side' island: texel → (around_mm at the local radius, arc_mm, r_mm, z_mm). prof in mm."""
    acc = [0.0]
    for p, q in zip(prof, prof[1:]):
        acc.append(acc[-1] + float(np.hypot(q[0] - p[0], q[1] - p[1])))
    u0, v0, u1, v1 = rect
    s = (v - v0) / (v1 - v0) * acc[-1]
    r = np.interp(s, acc, [p[0] for p in prof])
    z = np.interp(s, acc, [p[1] for p in prof])
    th = (u - u0) / (u1 - u0) * 2 * np.pi
    return th, s, r, z


def disc_map(u, v, rect, rmax, flip=False):
    u0, v0, u1, v1 = rect
    x = ((u - u0) / (u1 - u0) - 0.5) * 2 * rmax * (-1 if flip else 1)
    y = ((v - v0) / (v1 - v0) - 0.5) * 2 * rmax
    return x, y
