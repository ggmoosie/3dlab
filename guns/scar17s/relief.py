"""relief.py - surface detail measured on the profile master (right side), as 2-D masks in photo px -> signed-distance fields:
  hardware : black steel parts sitting proud of the FDE surfaces (port frame, deflector, QD plates, hinge plate, sockets)
  seams    : thin dark lines inside the FDE regions (panel lines, part seams, knuckle gaps) -> engraved grooves
  stipple  : high-local-variance texture regions (grip panels, vertical grip, magwell front) -> stippled panels
Logos and lettering are masked out (no markings are modelled). Shapes are measured, nothing from the photo is baked into textures."""
import numpy as np
from scipy import ndimage
from PIL import Image
import ref as R
from lab import sdf as S

# areas never used: FN logos, selector letters, stock length numbers, grip marking
NO_MARK = [((685, 1158), 24), ((1408, 1082), 22), ((754, 955), 16), ((700, 1028), 14)]
NO_MARK_BOX = [(260, 930, 455, 960)]
# black hardware that sits proud of the receiver (traced outlines); screws are modelled separately as Torx heads
HW_BOXES = [(616, 845, 634, 1000),            # hinge rear plate (black strip)
            (1416, 850, 1464, 895)]           # sling-loop cup
LOWER_HW_BOXES = [(696, 950, 766, 992),         # ambi selector lever + hub
                  (798, 954, 874, 980),         # bolt-release lever
                  (834, 966, 884, 1014)]        # mag-release button (inside its bezel)
STIPPLE_BOXES = [("grip", None), ("vgrip", None), (None, (1014, 975, 1048, 1048))]


def _no_mark(shape):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    m = np.zeros(shape, bool)
    for (cx, cy), r in NO_MARK:
        m |= (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    for x0, y0, x1, y1 in NO_MARK_BOX:
        m[y0:y1, x0:x1] = True
    return m


_M = {}


def masks():
    if _M:
        return _M
    a = R.image(); fg, dark = R.masks()
    L = a.mean(-1)
    nm = _no_mark(fg.shape)
    tan = fg & ~dark
    inner = ndimage.binary_erosion(fg, iterations=6)
    # hardware: dark pixels inside the hardware boxes (holes kept: screw recesses / windows read lighter)
    hb = np.zeros_like(fg)
    for x0, y0, x1, y1 in HW_BOXES:
        hb[y0:y1, x0:x1] = True
    hw = ndimage.binary_opening(dark & hb, iterations=1)
    hw = ndimage.binary_closing(hw, iterations=1)
    lb = np.zeros_like(fg)
    for x0, y0, x1, y1 in LOWER_HW_BOXES:
        lb[y0:y1, x0:x1] = True
    lhw = ndimage.binary_closing(dark & lb, iterations=1)
    # seams: black-hat thin dark lines anywhere on the gun surface, away from the outline and the hardware
    bh = ndimage.grey_closing(L, size=(5, 5)) - L
    seam = (bh > 18) & ndimage.binary_erosion(fg, iterations=3) & ~nm & ~ndimage.binary_dilation(hw, iterations=2)
    seam &= ~ndimage.binary_dilation(dark & ~ndimage.binary_erosion(dark, iterations=2), iterations=1) | ~dark
    lab, n = ndimage.label(seam, structure=np.ones((3, 3)))
    sz = ndimage.sum(seam, lab, range(1, n + 1))
    seam = np.isin(lab, 1 + np.nonzero(sz >= 14)[0])
    # stipple: local std of luminance on the grips / magwell front only
    m1 = ndimage.uniform_filter(L, 7); m2 = ndimage.uniform_filter(L * L, 7)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    reg = np.zeros_like(fg)
    for part, box in STIPPLE_BOXES:
        if part:
            reg |= R.poly_mask(R.PARTS[part], fg.shape)
        else:
            x0, y0, x1, y1 = box; reg[y0:y1, x0:x1] = True
    st = (sd > 11) & tan & inner & reg & ~nm
    st[:1078, 600:800] = False                                    # the smooth top of the pistol grip is not stippled
    st = ndimage.binary_closing(st, iterations=3); st = ndimage.binary_opening(st, iterations=3)
    lab, n = ndimage.label(st)
    sz = ndimage.sum(st, lab, range(1, n + 1))
    st = np.isin(lab, 1 + np.nonzero(sz >= 300)[0])
    st = ndimage.binary_fill_holes(st)
    _M.update(lower_hardware=lhw, hardware=hw, seam=seam & ~st, stipple=st | (nm & reg & ndimage.binary_erosion(fg, iterations=6)))
    return _M


_F = {}
SMOOTH = dict(hardware=1.6, lower_hardware=1.2)                    # px sigma; stipple stays as traced (bake uses it)


def field(name, grow_px=0.0):
    """Field2D (y, z mm) of a relief mask; seams are widened to a 0.55 mm groove"""
    key = (name, grow_px)
    if key in _F:
        return _F[key]
    m = masks()[name]
    if grow_px:
        m = ndimage.binary_dilation(m, iterations=int(grow_px))
    ys, xs = np.nonzero(m)
    x0, x1, r0, r1 = xs.min() - 10, xs.max() + 11, ys.min() - 10, ys.max() + 11
    c = m[r0:r1, x0:x1].astype(np.float32)
    up = 3
    im = Image.fromarray((c * 255).astype(np.uint8)).resize(((x1 - x0) * up, (r1 - r0) * up), Image.BILINEAR)
    a = np.asarray(im).astype(np.float32) / 255
    if SMOOTH.get(name):                                          # traced pixel stairs -> clean machined outlines
        a = ndimage.gaussian_filter(a, SMOOTH[name] * up)
    hi = a > 0.5
    res = R.S_MM / up
    d = (ndimage.distance_transform_edt(~hi) - ndimage.distance_transform_edt(hi)) * res
    y0, _ = R.px2b(x0, 0); _, z1 = R.px2b(0, r0)
    f = S.Field2D(d.astype(np.float32), float(y0), float(z1), res, outside=20.0)
    _F[key] = f
    return f


def preview(out):
    a = R.image().copy(); M = masks()
    for k, c in (("hardware", (0, 160, 255)), ("seam", (255, 0, 0)), ("stipple", (0, 220, 80))):
        a[M[k]] = a[M[k]] * 0.3 + np.array(c) * 0.7
    Image.fromarray(a.astype(np.uint8)).crop((150, 730, 1890, 1240)).save(out)
