"""overlay.py - feature spot-the-difference: the reference laid OVER the model through the matched camera, per region.
Tiles: photo | model | overlay (model shading, photo's interior edges in red, model's edges in cyan). A red line with
no cyan partner = a feature the photo has and the model lacks (a seam, a pin, a lever, a step). Uses compare.py's
renders + fitted camera for <tag>. Output (contains photos -> work/): work/overlay_<tag>_<side>.jpg
usage: .venv/bin/python guns/ak74m/overlay.py tag=v6 [side=LR] [only=receiver]"""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
sys.argv = sys.argv[:1]
import compare as C
from regions import REGIONS

TAG = A.get("tag", "v6")


def edges(g, sigma, q):
    g = ndimage.gaussian_filter(g.astype(np.float32), sigma)
    m = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    return m > np.percentile(m, q)


def main():
    cams = json.load(open(os.path.join(HERE, "review", f"{TAG}_iou.json")))
    only = A["only"].split(",") if "only" in A else None
    for side in A.get("side", "LR"):
        cam = cams[side]["camera"]
        ph = np.asarray(Image.open(C.SIDES[side]["photo"]).convert("RGB")).astype(np.float32)
        photo = C.warp(ph, *cam.values()).clip(0, 255).astype(np.uint8)
        mod = np.asarray(Image.open(os.path.join(C.WORK, f"cmp_{TAG}_{side}_b.png")).convert("RGB"))
        al = np.asarray(Image.open(os.path.join(C.WORK, f"cmp_{TAG}_{side}_a.png")))[..., 3] > 127
        inner = ndimage.binary_erosion(al, iterations=2)
        ep = edges(photo.mean(-1), 0.8, 88) & inner                    # interior photo detail only
        em = edges(mod.mean(-1), 0.8, 90) & inner
        g = (mod.mean(-1) * 0.55 + 100).clip(0, 255).astype(np.uint8)
        ov = np.stack([g] * 3, -1)
        ov[em] = (40, 200, 230); ov[ep] = (235, 40, 40); ov[ep & ndimage.binary_dilation(em, iterations=1)] = (250, 250, 250)
        rows = []
        for name, box in REGIONS:
            if only and not any(o in name for o in only):
                continue
            if side == "R":                                             # R camera frame is mirrored: muzzle on the right
                W = photo.shape[1] + 6; box = (W - box[2], box[1] + 10, W - box[0], box[3] + 10)
            w, h = box[2] - box[0], box[3] - box[1]
            k = max(1, int(round(min(520 / w, 300 / h))))
            tiles = [Image.fromarray(t).crop(box).resize((w * k, h * k), Image.LANCZOS) for t in (photo, mod)]
            tiles.append(Image.fromarray(ov).crop(box).resize((w * k, h * k), Image.NEAREST))
            rows.append((name, tiles))
        Wt = max(sum(t.width for t in ts) + 20 for _, ts in rows); Ht = sum(ts[0].height + 24 for _, ts in rows) + 30
        sh = Image.new("RGB", (Wt, Ht), (30, 30, 30)); d = ImageDraw.Draw(sh)
        d.text((8, 8), f"AK-74M feature overlay {TAG} {side}: photo | model | overlay (red photo edge only = missing feature, cyan model only, white both)", fill="white")
        y = 30
        for lab, ts in rows:
            d.text((8, y + 3), lab, fill=(255, 220, 120)); y += 20; x = 0
            for t in ts:
                sh.paste(t, (x, y)); x += t.width + 10
            y += ts[0].height + 4
        out = os.path.join(C.WORK, f"overlay_{TAG}_{side}.jpg"); sh.save(out, quality=88); print("OVERLAY", out, sh.size)


if __name__ == "__main__":
    main()
