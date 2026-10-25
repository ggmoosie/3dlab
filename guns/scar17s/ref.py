"""ref.py - the profile master: the user's near-orthographic right-side photo of a SCAR 17S NRCH (refs_user/nrch_right.jpg,
never committed). Calibrated on the 1913 top rail: slot pitch 17.86 px = 10.01 mm -> S = 0.5605 mm/px. Bore row 920.5 (barrel
centre), muzzle face at x 1873 (= MUZZLE_Y in the B frame), which puts the bolt face at x 1047 = the front of the ejection port.
Photo x grows toward the muzzle, so B y = (x - X_MUZ) * S + MUZZLE_Y and B z = (ROW_BORE - row) * S.
prof(name) -> Field2D signed distance (mm, <0 inside) of the photo silhouette cut by a part's partition polygon (px)."""
import os, numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S

PHOTO = os.path.join(HERE, "refs_user", "nrch_right.jpg")
S_MM = 10.01 / 17.857
X_MUZ, ROW_BORE = 1873.0, 920.5
MUZZLE_Y = 463.0
UP = 3                                     # supersampling of the traced masks


def px2b(x, row):
    return (np.asarray(x, float) - X_MUZ) * S_MM + MUZZLE_Y, (ROW_BORE - np.asarray(row, float)) * S_MM


def b2px(y, z):
    return (np.asarray(y, float) - MUZZLE_Y) / S_MM + X_MUZ, ROW_BORE - np.asarray(z, float) / S_MM


_IMG = None


def image():
    global _IMG
    if _IMG is None:
        _IMG = np.asarray(Image.open(PHOTO).convert("RGB")).astype(np.float32)
    return _IMG


_MASKS = None


def masks():
    """foreground (anything not paper-white), dark (black steel / rubber), tan (FDE); computed once"""
    global _MASKS
    if _MASKS is not None:
        return _MASKS
    a = image()
    mn, mx = a.min(-1), a.max(-1)
    fg = ~((mn > 228) & (mx - mn < 18))
    fg = ndimage.binary_opening(fg, iterations=1)
    lab, n = ndimage.label(fg)
    size = ndimage.sum(fg, lab, range(1, n + 1))
    fg = np.isin(lab, 1 + np.nonzero(size > 400)[0])
    dark = fg & (mx < 105) & (mx - mn < 40)
    _MASKS = (fg, dark)
    return _MASKS


# partition polygons in photo px (generous: the silhouette decides the outline)
PARTS = {
    "rail":    [(626, 824), (1602, 824), (1602, 846), (626, 846)],
    "upper":   [(600, 843), (1606, 843), (1606, 968), (1048, 968), (1048, 945), (600, 945)],
    "hinge":   [(572, 840), (626, 840), (626, 1002), (572, 1002)],
    "lower":   [(620, 943), (1048, 943), (1050, 968), (1050, 1052), (866, 1052), (864, 1080), (756, 1080), (748, 1000), (620, 1000)],
    "grip":    [(632, 990), (772, 990), (776, 1215), (620, 1215)],
    "mag":     [(866, 1050), (1016, 1050), (1016, 1230), (866, 1230)],
    "stock":   [(206, 840), (574, 840), (574, 1130), (206, 1130)],
    "butt":    [(168, 870), (207, 870), (207, 1130), (168, 1130)],
    "vgrip":   [(1325, 966), (1462, 966), (1462, 1215), (1325, 1215)],
    "rsight":  [(648, 750), (728, 750), (728, 838), (648, 838)],
    "fsight":  [(1512, 750), (1584, 750), (1584, 846), (1512, 846)],
    "trigger": [(800, 1002), (848, 1002), (848, 1074), (800, 1074)],
}
DARK_ONLY = {"fsight", "trigger"}


def poly_mask(pts, shape):
    im = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon(pts, fill=255)
    return np.asarray(im) > 127


_FIELDS = {}


BLUR = dict(stock=1.4, butt=1.0)


def prof(name, grow=0.0):
    """signed-distance Field2D (y, z mm) of the silhouette inside the part's polygon; grow (mm) inflates it"""
    if name in _FIELDS:
        return _FIELDS[name]
    fg, dark = masks()
    m = fg & poly_mask(PARTS[name], fg.shape)
    if name in DARK_ONLY:
        m &= ndimage.binary_dilation(dark, iterations=2)
        m = ndimage.binary_closing(m, iterations=2)
    ys, xs = np.nonzero(m)
    x0, x1, r0, r1 = xs.min() - 12, xs.max() + 13, ys.min() - 12, ys.max() + 13
    c = m[r0:r1, x0:x1].astype(np.float32)
    im = Image.fromarray((c * 255).astype(np.uint8)).resize(((x1 - x0) * UP, (r1 - r0) * UP), Image.BILINEAR)
    im = im.filter(ImageFilter.GaussianBlur(UP * BLUR.get(name, 0.6)))           # jpeg edge noise -> clean outline
    hi = np.asarray(im) > 127
    res = S_MM / UP
    d = (ndimage.distance_transform_edt(~hi) - ndimage.distance_transform_edt(hi)) * res
    # Field2D wants row 0 = top (max z), col 0 = min y
    y0, _ = px2b(x0, 0)
    _, z1 = px2b(0, r0)
    f = S.Field2D(d.astype(np.float32), float(y0), float(z1), res, outside=50.0)
    _FIELDS[name] = f
    return f


def extent(name):
    """part bbox in B mm: (y0, y1, z0, z1)"""
    fg, dark = masks()
    m = fg & poly_mask(PARTS[name], fg.shape)
    if name in DARK_ONLY:
        m &= dark
    ys, xs = np.nonzero(m)
    (ya, za), (yb, zb) = px2b(xs.min(), ys.max()), px2b(xs.max(), ys.min())
    return float(ya), float(yb), float(za), float(zb)


def column_radius(x0, x1):
    """lathe radius per photo column (barrel, brake): half the silhouette height around the bore row"""
    fg, _ = masks()
    out = []
    for x in range(int(x0), int(x1) + 1):
        col = np.nonzero(fg[860:990, x])[0] + 860
        if len(col):
            out.append((float(px2b(x, 0)[0]), (col.max() - col.min() + 1) * S_MM / 2))
    return out
