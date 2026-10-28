"""ak_ref.py - profile master: the user's near-orthographic LEFT-side photo of the AK-74M (refs_user/...163245.png, private).
Muzzle on the LEFT of the photo. Calibrated to the 943 mm overall length: muzzle face x 154 -> butt rear x 1152 = 998 px,
S = 0.9449 mm/px. Bore row 277.5 at the muzzle. A small roll ROT (deg) about the muzzle point is allowed (fit by silcmp.py).
B frame: y = (X_MUZ - x') * S (muzzle face y = 0, gun at y < 0), z = (ROW_BORE - row') * S, (x', row') = photo px un-rolled.
prof(name) -> Field2D (y, z) signed distance (mm, < 0 inside) of the photo silhouette cut by the part's partition polygon."""
import os, sys, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S

PHOTO = os.path.join(HERE, "refs_user", "Screenshot 2026-10-25 163245.png")
PHOTO_R = os.path.join(HERE, "refs_user", "Screenshot 2026-10-25 163254.png")
X_MUZ, ROW_BORE = 154.0, 277.5
S_MM = 943.0 / 998.0
ROT = float(os.environ.get("AK_ROT", "0.0"))          # photo roll (deg, + = muzzle up in the photo)
UP = 3


def _unroll(x, row):
    a = math.radians(ROT); c, s = math.cos(a), math.sin(a)
    dx, dr = np.asarray(x, float) - X_MUZ, np.asarray(row, float) - ROW_BORE
    return X_MUZ + c * dx - s * dr, ROW_BORE + s * dx + c * dr


def px2b(x, row):
    xu, ru = _unroll(x, row)
    return (X_MUZ - xu) * S_MM, (ROW_BORE - ru) * S_MM


def b2px(y, z):
    xu, ru = X_MUZ - np.asarray(y, float) / S_MM, ROW_BORE - np.asarray(z, float) / S_MM
    a = math.radians(-ROT); c, s = math.cos(a), math.sin(a)
    dx, dr = xu - X_MUZ, ru - ROW_BORE
    return X_MUZ + c * dx - s * dr, ROW_BORE + s * dx + c * dr


def bp(x, row):
    y, z = px2b(x, row)
    return float(y), float(z)


_M = {}


def masks(photo=PHOTO):
    """foreground (not paper white), largest component + its near neighbours"""
    if photo in _M:
        return _M[photo]
    a = np.asarray(Image.open(photo).convert("RGB")).astype(np.float32)
    mn, mx = a.min(-1), a.max(-1)
    fg = ~((mn > 222) & (mx - mn < 20))
    fg = ndimage.binary_opening(fg, iterations=1)
    lab, n = ndimage.label(fg)
    sz = ndimage.sum(fg, lab, range(1, n + 1))
    fg = np.isin(lab, 1 + np.nonzero(sz > 150)[0])
    holes = ndimage.binary_fill_holes(fg) & ~fg                                       # enclosed background
    hl, hn = ndimage.label(holes)
    hs = ndimage.sum(holes, hl, range(1, hn + 1))
    fg = fg | np.isin(hl, 1 + np.nonzero(hs < 90)[0])                                  # small = shiny pin heads, not holes
    _M[photo] = (fg, a)
    return _M[photo]


# partition polygons, photo px (generous; the silhouette decides the outline)
PARTS = {
    "brake":    [(148, 258), (241, 258), (241, 296), (148, 296)],
    "fsb":      [(238, 214), (279, 214), (279, 306), (238, 306)],
    "barrel":   [(276, 266), (565, 266), (565, 289), (276, 289)],
    "rod":      [(244, 288), (560, 288), (560, 298), (244, 298)],
    "gasblock": [(338, 252), (375, 252), (375, 302), (338, 302)],
    "gastube":  [(352, 234), (448, 234), (448, 264), (352, 264)],
    "hg_upper": [(444, 230), (565, 230), (565, 262), (444, 262)],
    "hg_lower": [(444, 262), (575, 262), (575, 320), (444, 320)],
    "rsight":   [(564, 212), (660, 212), (660, 252), (564, 252)],
    "dcover":   [(605, 230), (892, 230), (892, 262), (605, 262)],
    "receiver": [(560, 250), (892, 250), (892, 347), (828, 347), (828, 304), (560, 304)],
    "tguard":   [(760, 300), (828, 300), (828, 352), (760, 352)],              # parametric strip (thin, light: not traced)
    "trigger":  [(781, 303), (812, 303), (812, 339), (781, 339)],
    "magcatch": [(731, 300), (769, 300), (769, 362), (731, 362)],
    "grip":     [(836, 318), (935, 300), (935, 425), (836, 425)],
    "mag":      [(596, 300), (732, 300), (732, 498), (596, 498)],
    "stock":    [(886, 252), (1158, 252), (1158, 385), (886, 385)],
}


def poly_mask(poly, shape):
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in poly], fill=255)
    return np.asarray(im) > 127


_F = {}


def prof(name, blur=0.6, photo=PHOTO):
    """Field2D (y, z) of the silhouette inside PARTS[name]; supersampled + lightly blurred (jpeg edge noise)"""
    key = (name, photo)
    if key in _F:
        return _F[key]
    fg, _ = masks(photo)
    m = fg & poly_mask(PARTS[name], fg.shape)
    ys, xs = np.nonzero(m)
    x0, x1, r0, r1 = xs.min() - 10, xs.max() + 11, ys.min() - 10, ys.max() + 11
    c = Image.fromarray((m[r0:r1, x0:x1] * 255).astype(np.uint8)).resize(((x1 - x0) * UP, (r1 - r0) * UP), Image.BILINEAR)
    c = c.filter(ImageFilter.GaussianBlur(UP * blur))
    hi = np.asarray(c) > 127
    res = S_MM / UP
    d = (ndimage.distance_transform_edt(~hi) - ndimage.distance_transform_edt(hi)) * res
    # Field2D wants columns increasing in its first coordinate: our y DEcreases with photo x -> flip columns
    d = d[:, ::-1]
    y0 = (X_MUZ - x1) * S_MM
    z1 = (ROW_BORE - r0) * S_MM
    f = S.Field2D(d.astype(np.float32), float(y0), float(z1), res, outside=50.0)
    _F[key] = f
    return f


def extent(name):
    """part bbox in B mm (y0, y1, z0, z1) from its silhouette"""
    fg, _ = masks()
    m = fg & poly_mask(PARTS[name], fg.shape)
    ys, xs = np.nonzero(m)
    (ya, za), (yb, zb) = px2b(xs.max(), ys.max()), px2b(xs.min(), ys.min())
    return float(ya), float(yb), float(za), float(zb)


def column_radius(x0, x1, name):
    """per photo column inside part `name`: (y_mm, half-height about the bore row, centre z) of the silhouette"""
    fg, _ = masks()
    m = fg & poly_mask(PARTS[name], fg.shape)
    out = []
    for x in range(int(x0), int(x1) + 1):
        r = np.nonzero(m[:, x])[0]
        if len(r):
            y, za = px2b(x, r.max() + 0.5); _, zb = px2b(x, r.min() - 0.5)
            out.append((y, (zb - za) / 2, (zb + za) / 2))
    return out
