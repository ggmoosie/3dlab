"""compare.py - S5 gate against the profile master: an orthographic camera locked to the photo calibration (ref.py), a flat
silhouette render -> IoU vs the photo silhouette, plus a beauty render for photo | model | overlay | difference sheets.
usage: .venv/bin/python guns/scar17s/compare.py [src=high|game] [samples=32] [tag=v2]   exit 3 if IoU < 0.95"""
import bpy, os, sys, math, json
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import scene
import ref as R
import bl_high
import scar_spec as P

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
SRC, SMP, TAG = ARGS.get("src", "high"), int(ARGS.get("samples", 32) or 32), ARGS.get("tag", "v2")
REV = os.path.join(HERE, "review")
W = 2000


def cam_photo():
    yb, zb = R.px2b(1000, 1000)
    o = np.array(P.ORIGIN_B)
    c = Vector((0.0, -(yb - o[1]) * 0.001, (zb - o[2]) * 0.001))
    cd = bpy.data.cameras.new("photo"); cd.type = "ORTHO"; cd.ortho_scale = W * R.S_MM * 0.001
    cd.clip_start, cd.clip_end = 0.01, 10
    cam = bpy.data.objects.new("photo", cd); bpy.context.scene.collection.objects.link(cam)
    cam.location = c + Vector((-2.0, 0, 0))
    cam.rotation_euler = Vector((1, 0, 0)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def load_objs():
    if SRC == "high":
        objs = list(bl_high.load().values())
    else:
        bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "scar17s.blend"))
        objs = [o for o in bpy.data.objects if o.name.startswith("SM_SCAR17S_") and "_LOD" not in o.name and o.type == "MESH"]
        for o in bpy.data.objects:
            if o not in objs:
                o.hide_render = True
    return objs


def silhouette(path):
    """alpha of a 1-spp Cycles render with a transparent film = the model's silhouette (no GPU needed)"""
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.device = "CPU"; s.cycles.samples = 1; s.cycles.use_denoising = False
    s.cycles.filter_width = 0.5
    s.render.film_transparent = True
    s.render.resolution_x = s.render.resolution_y = W
    s.render.image_settings.color_mode = "RGBA"
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return np.asarray(Image.open(path))[..., 3] > 127


def main():
    if SRC == "high":
        scene.reset()
    objs = load_objs()
    for o in objs:
        if o.name.endswith("round") or o.name.endswith("Round"):
            o.hide_render = True
    cam_photo()
    os.makedirs(REV, exist_ok=True)
    sil = silhouette(os.path.join(REV, f"cmp_{TAG}_sil.png"))
    fg, _ = R.masks()
    inter, uni = (sil & fg).sum(), (sil | fg).sum()
    iou = inter / uni
    only_photo, only_model = (fg & ~sil).sum() / fg.sum(), (sil & ~fg).sum() / fg.sum()
    # beauty: studio lights, white background like the photo
    scene.studio(objs, backdrop=False)
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.samples = SMP; s.cycles.use_denoising = True
    s.render.film_transparent = True
    s.view_settings.view_transform = "AgX"; s.view_settings.look = "AgX - Punchy"; s.view_settings.exposure = 0.2
    s.render.resolution_x = s.render.resolution_y = W
    s.render.filepath = os.path.join(REV, f"cmp_{TAG}_beauty.png")
    bpy.ops.render.render(write_still=True)
    bt = Image.open(s.render.filepath).convert("RGBA")
    white = Image.new("RGBA", bt.size, (255, 255, 255, 255)); white.alpha_composite(bt)
    model = np.asarray(white.convert("RGB")).astype(np.float32)
    photo = R.image()
    crop = (150, 720, 1890, 1240)
    ov = (photo * 0.5 + model * 0.5)
    diff = np.full_like(photo, 245.0)
    diff[fg & sil] = (200, 200, 200)
    diff[fg & ~sil] = (220, 40, 40)
    diff[sil & ~fg] = (30, 190, 220)
    tiles = [("photo (profile master)", photo), ("model, same camera", model), ("50 % overlay", ov),
             (f"difference  IoU {iou:.3f}   red = photo only {only_photo * 100:.1f} %   cyan = model only {only_model * 100:.1f} %", diff)]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    sheet = Image.new("RGB", (cw, (ch + 40) * 4 + 60), (22, 23, 26))
    d = ImageDraw.Draw(sheet)
    fdir = os.path.join(HERE, "../../fonts")
    fb = ImageFont.truetype(os.path.join(fdir, "WorkSans-Bold.ttf"), 30); fs = ImageFont.truetype(os.path.join(fdir, "WorkSans-Bold.ttf"), 22)
    d.text((14, 12), f"SCAR 17S NRCH vs the reference photo - right side, matched orthographic camera ({TAG}, {SRC})", font=fb, fill=(235, 235, 235))
    for i, (lab, a) in enumerate(tiles):
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).crop(crop)
        y = 60 + i * (ch + 40)
        d.text((14, y + 6), lab, font=fs, fill=(230, 230, 230))
        sheet.paste(im, (0, y + 38))
    out = os.path.join(REV, f"compare_{TAG}_{SRC}.jpg")
    sheet.save(out, quality=90)
    json.dump(dict(iou=float(iou), photo_only=float(only_photo), model_only=float(only_model)), open(os.path.join(REV, f"compare_{TAG}_{SRC}.json"), "w"))
    print(f"COMPARE IoU {iou:.4f} photo_only {only_photo:.4f} model_only {only_model:.4f} -> {out}")
    print("COMPARE VERDICT", "PASS" if iou >= 0.95 else "FAIL")
    if iou < 0.95:
        sys.exit(3)


if __name__ == "__main__":
    main()
