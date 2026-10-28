"""regions.py - front-to-back region sheets: for each region of the profile, the left photo | the right photo (mirrored +
aligned, work/ref_R_aligned.png) | the model through the matched left camera | the silhouette diff. Uses compare.py's
renders for <tag>. Output (contains photos -> work/): work/regions_<tag>.jpg
usage: .venv/bin/python guns/ak74m/regions.py tag=v3 [only=brake,fsb]"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ak_ref as R

A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
TAG = A.get("tag", "v3")
REGIONS = [("muzzle brake", (146, 255, 245, 300)), ("front sight + gas block", (232, 212, 380, 306)),
           ("gas tube + handguards", (340, 225, 580, 325)), ("rear sight + dust cover front", (555, 205, 700, 270)),
           ("receiver side", (560, 240, 895, 352)), ("trigger group", (720, 290, 900, 370)),
           ("pistol grip", (820, 295, 945, 425)), ("magazine", (590, 290, 740, 500)), ("stock + hinge", (880, 245, 1160, 390))]


def main():
    only = A["only"].split(",") if "only" in A else None
    ph = Image.open(R.PHOTO).convert("RGB")
    pr = Image.open(os.path.join(HERE, "work", "ref_R_aligned.png")).convert("RGB")
    md = Image.open(os.path.join(HERE, "work", f"cmp_{TAG}_L_b.png")).convert("RGB")
    fg, _ = R.masks()
    al = np.asarray(Image.open(os.path.join(HERE, "work", f"cmp_{TAG}_L_a.png")))[..., 3] > 127
    df = np.full(fg.shape + (3,), 245, np.uint8); df[al & fg] = 150; df[fg & ~al] = (230, 40, 40); df[al & ~fg] = (40, 200, 230)
    df = Image.fromarray(df)
    rows = []
    for name, box in REGIONS:
        if only and not any(o in name for o in only):
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        k = max(1, int(round(min(380 / w, 260 / h))))
        tiles = [im.crop(box).resize((w * k, h * k), Image.LANCZOS) for im in (ph, pr, md)] + [df.crop(box).resize((w * k, h * k), Image.NEAREST)]
        m = np.asarray(fg)[box[1]:box[3], box[0]:box[2]]; a = al[box[1]:box[3], box[0]:box[2]]
        iou = (m & a).sum() / max((m | a).sum(), 1)
        rows.append((f"{name}  (region IoU {iou:.3f})", tiles)); print(f"REGION {name:32s} {iou:.3f}")
    W = max(sum(t.width for t in ts) + 30 for _, ts in rows)
    H = sum(ts[0].height + 26 for _, ts in rows) + 30
    sh = Image.new("RGB", (W, H), (30, 30, 30)); d = ImageDraw.Draw(sh)
    d.text((8, 8), f"AK-74M front-to-back regions, pass {TAG}: left photo | right photo (aligned) | model, matched camera | diff (red photo only, cyan model only)", fill="white")
    y = 30
    for lab, ts in rows:
        d.text((8, y + 4), lab, fill=(255, 220, 120)); x = 0; y += 22
        for t in ts:
            sh.paste(t, (x, y)); x += t.width + 10
        y += ts[0].height + 4
    out = os.path.join(HERE, "work", f"regions_{TAG}.jpg"); sh.save(out, quality=88); print("REGIONS", out)


if __name__ == "__main__":
    main()
