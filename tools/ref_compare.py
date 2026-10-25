"""ref_compare.py - "spot the difference" sheet: reference photo vs a render from the matched camera (camera_match.py).

    python ref_compare.py <reference image> <render.png (RGBA, same camera)> <out_prefix> [--crop x0,y0,x1,y1]

Writes <out_prefix>_sheet.jpg with four panels at the photo's own scale:
  1 reference | 2 render | 3 50 % overlay | 4 silhouette + edge difference
Panel 4: RED = only in the photo (model too small / missing feature), CYAN = only in the render (model too big / extra),
WHITE edges = photo edges, ORANGE edges = render edges; where they do not coincide a feature is misplaced or missing.
The silhouette IoU is printed and written on the sheet when the photo has a plain background (product shot); on a busy
background only the overlay and the edges are meaningful. --crop zooms all four panels to one region (do this for every
area that matters: the first-person view surfaces especially).
Then WRITE the list of differences (feature, where, which way it is off, by how much) and fix every one before re-rendering.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageChops

a = [x for x in sys.argv[1:] if not x.startswith('--')]
ref_p, ren_p, out = a[0], a[1], a[2]
crop = None
if '--crop' in sys.argv: crop = tuple(int(v) for v in sys.argv[sys.argv.index('--crop') + 1].split(','))

ref = Image.open(ref_p).convert('RGB'); ren = Image.open(ren_p).convert('RGBA')
if ren.size != ref.size: ren = ren.resize(ref.size, Image.LANCZOS)          # camera_match renders at the photo size; this is a fallback
R = np.asarray(ref).astype(np.float32); A = np.asarray(ren).astype(np.float32)

# reference silhouette: distance from the border colour (works for studio/product shots on a plain background)
# background = the dominant colour of a band just inside the edges (photos often carry a thin frame line or a watermark at the very edge)
hh, ww = R.shape[:2]; i0 = max(2, int(0.015 * min(hh, ww))); i1 = i0 + 6
border = np.concatenate([R[i0:i1].reshape(-1, 3), R[-i1:-i0].reshape(-1, 3), R[:, i0:i1].reshape(-1, 3), R[:, -i1:-i0].reshape(-1, 3)])
q = (border // 8).astype(np.int32); keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
mode = np.bincount(keys).argmax(); dom = border[keys == mode]
bg = np.median(dom, 0); plain = len(dom) / len(border) > 0.6 and float(dom.std(0).mean()) < 12.0
dist = np.linalg.norm(R - bg, axis=2)
def otsu(v):
    h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    btw = (m[-1] * w / w[-1] - m) ** 2 / np.maximum(w * (w[-1] - w), 1); return c[np.argmax(btw)]
thr = float(sys.argv[sys.argv.index('--bg-thresh') + 1]) if '--bg-thresh' in sys.argv else max(8.0, min(60.0, otsu(dist)))
ref_mask = dist > thr
ren_mask = A[..., 3] > 127
ref_mask_img = Image.fromarray((ref_mask * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
ref_mask = np.asarray(ref_mask_img) > 127
iou = (ref_mask & ren_mask).sum() / max(1, (ref_mask | ren_mask).sum())

comp = Image.new('RGB', ref.size, tuple(int(v) for v in bg)) if plain else Image.new('RGB', ref.size, (128, 128, 128))
comp.paste(ren, (0, 0), ren)                                               # the render on the photo's background
overlay = Image.blend(ref, comp, 0.5)

diff = np.full(R.shape, 40, np.uint8)
if plain:
    diff[ref_mask & ~ren_mask] = (230, 40, 40); diff[ren_mask & ~ref_mask] = (40, 210, 230); diff[ref_mask & ren_mask] = (70, 70, 70)
e_ref = np.asarray(ref.convert('L').filter(ImageFilter.FIND_EDGES)) > 40
e_ren = np.asarray(comp.convert('L').filter(ImageFilter.FIND_EDGES)) > 40
diff[e_ref] = (255, 255, 255); diff[e_ren & ~e_ref] = (255, 150, 30)
diffi = Image.fromarray(diff)

panels = [('reference', ref), ('render (matched camera)', comp), ('50% overlay', overlay),
          ('diff: red=photo only, cyan=render only, white/orange=photo/render edges' + (' | IoU %.3f' % iou if plain else ' | busy bg: no IoU'), diffi)]
if crop: panels = [(t, im.crop(crop)) for t, im in panels]
w, h = panels[0][1].size; s = min(1.0, 1400.0 / w); w2, h2 = int(w * s), int(h * s)
sheet = Image.new('RGB', (w2 * 2 + 10, (h2 + 28) * 2 + 10), (25, 25, 28)); d = ImageDraw.Draw(sheet)
for i, (t, im) in enumerate(panels):
    x, y = (i % 2) * (w2 + 10), (i // 2) * (h2 + 38)
    d.text((x + 6, y + 6), t, fill=(255, 200, 90)); sheet.paste(im.resize((w2, h2), Image.LANCZOS), (x, y + 28))
sheet.save(out + '_sheet.jpg', quality=92)
print('COMPARE iou %s plain_bg %s -> %s_sheet.jpg' % (('%.3f' % iou) if plain else 'n/a', plain, out))

# the silhouette gate is enforced on plain-background photos (IoU is over the whole image, --crop only zooms the sheet):
# below --min-iou (default 0.95, the S2 blockout gate; use 0.97 for final product-shot passes, 0 to only report) exits 3.
# On a busy background there is no IoU and the written difference list is the gate.
min_iou = float(sys.argv[sys.argv.index('--min-iou') + 1]) if '--min-iou' in sys.argv else 0.95
if not plain:
    print('COMPARE VERDICT n/a (busy background: no silhouette IoU; the written difference list is the gate)')
elif iou < min_iou:
    print('COMPARE VERDICT FAIL: IoU %.3f is under %.3f' % (iou, min_iou)); sys.exit(3)
else:
    print('COMPARE VERDICT PASS: IoU %.3f >= %.3f' % (iou, min_iou))
