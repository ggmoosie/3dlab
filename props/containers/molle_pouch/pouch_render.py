"""Renders for the GPPouch: hero 3/4, reference-angle matches (front / 3/4 / back / side), flap closed vs open, skins,
contact sheet.  .venv/bin/python props/containers/molle_pouch/pouch_render.py [--fast] [--only hero,refs,open,variants]

Reference-angle cameras are solved, not eyeballed: the game mesh's silhouette is projected and the camera (yaw, pitch,
roll, distance, lens, aim) is optimised for silhouette IoU against the private photo in refs_user/ (never copied into
renders/; the side-by-side comparison sheets go to work/molle_pouch/compare/, which is not committed)."""
import os, sys, math, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import bpy
import numpy as np
from mathutils import Vector, Matrix
from PIL import Image, ImageDraw, ImageFilter
from scipy.optimize import minimize
from lab import scene

FAST = "--fast" in sys.argv
SMP = 16 if FAST else 64
REND = os.path.join(HERE, "renders")
CMP = os.path.join(ROOT, "work/molle_pouch/compare")
REFD = os.path.join(HERE, "refs_user")
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else ["hero", "refs", "open", "closeup", "variants", "sheet"]
REFS = {"front": ("front.webp", 0, 6), "front34": ("front34.webp", 38, 22), "back": ("back.jpg", 180, 8), "side": ("side.jpg", 90, 4)}


def mesh_world(objs):
    V, T = [], []
    n = 0
    for o in objs:
        dg = bpy.context.evaluated_depsgraph_get()
        oe = o.evaluated_get(dg); me = oe.to_mesh(); me.calc_loop_triangles()
        M = np.array(o.matrix_world)
        co = np.array([v.co[:] for v in me.vertices])
        co = co @ M[:3, :3].T + M[:3, 3]
        V.append(co); T.append(np.array([t.vertices[:] for t in me.loop_triangles]) + n)
        n += len(co)
        oe.to_mesh_clear()
    return np.concatenate(V), np.concatenate(T)


def cam_basis(x, c):
    yaw, pitch, roll, dist, f, tx, tz, ty = x
    y, p = math.radians(yaw), math.radians(pitch)
    d = np.array([math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)])
    tgt = c + np.array([tx, ty, tz])
    pos = tgt + d * dist
    fw = -d
    rt = np.cross(fw, [0, 0, 1.0]); rt /= np.linalg.norm(rt)
    up = np.cross(rt, fw)
    r = math.radians(roll)
    rt, up = rt * math.cos(r) + up * math.sin(r), -rt * math.sin(r) + up * math.cos(r)
    return pos, fw, rt, up


def project(x, c, V, W, H):
    pos, fw, rt, up = cam_basis(x, c)
    q = V - pos
    z = q @ fw
    fpx = x[4] / 36.0 * W
    return np.stack([W / 2 + fpx * (q @ rt) / z, H / 2 - fpx * (q @ up) / z], 1)


def sil(x, c, V, T, W, H):
    uv = project(x, c, V, W, H)
    im = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(im)
    for t in uv[T]:
        d.polygon([tuple(p) for p in t], fill=255)
    return np.asarray(im) > 127


def photo_mask(path, W, H):
    R = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS)).astype(np.float32)
    dist = np.linalg.norm(R - np.array([255, 255, 255.0]), axis=2)
    m = Image.fromarray(((dist > 28) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    return np.asarray(m) > 127


def solve(name, V, T, c):
    fn, yaw, pitch = REFS[name]
    im = Image.open(os.path.join(REFD, fn))
    W0, H0 = im.size
    k = 360.0 / max(W0, H0)
    W, H = int(W0 * k), int(H0 * k)
    M = photo_mask(os.path.join(REFD, fn), W, H)
    ys, xs = np.nonzero(M)
    hpx = ys.max() - ys.min()
    f0 = 100.0
    dist0 = (f0 / 36 * W) * (V[:, 2].max() - V[:, 2].min()) / hpx
    cx_ph, cy_ph = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2

    def cost(x):
        S = sil(x, c, V, T, W, H)
        pen = max(0.0, 45.0 - x[4]) * 0.01 + max(0.0, x[4] - 400.0) * 0.01
        pen += max(0.0, abs(x[0] - yaw) - 8.0) * 0.02 + max(0.0, abs(x[1] - pitch) - 12.0) * 0.02 + abs(x[2]) * 0.01
        return -(S & M).sum() / max(1, (S | M).sum()) + pen
    x = np.array([yaw, pitch, 0.0, dist0, f0, 0.0, 0.0, 0.0])
    # aim: shift the target so the silhouette centres match before optimising
    for _ in range(2):
        S = sil(x, c, V, T, W, H)
        yy, xx = np.nonzero(S)
        du, dv = cx_ph - (xx.min() + xx.max()) / 2, cy_ph - (yy.min() + yy.max()) / 2
        pos, fw, rt, up = cam_basis(x, c)
        sc = x[3] / (x[4] / 36 * W)
        shift = -rt * du * sc + up * dv * sc
        x[5] -= shift[0]; x[6] -= shift[2]; x[7] -= shift[1]
    best = None
    for f in (60.0, 100.0, 160.0):
        x1 = x.copy(); x1[4] = f; x1[3] = x[3] * f / f0
        r = minimize(cost, x1, method="Nelder-Mead", options=dict(maxiter=500, xatol=1e-3, fatol=1e-5,
                     initial_simplex=None))
        if best is None or r.fun < best.fun:
            best = r
    print("SOLVE", name, "IoU %.3f" % -best.fun, "x", np.round(best.x, 4).tolist(), flush=True)
    return best.x, (W0, H0), -best.fun


def make_cam(x, c, name, lens_res):
    pos, fw, rt, up = cam_basis(x, c)
    cd = bpy.data.cameras.new(name)
    cd.lens = x[4]; cd.sensor_fit = "HORIZONTAL"; cd.sensor_width = 36.0; cd.clip_start = 0.001
    o = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(o)
    R = Matrix([list(rt), list(up), list(-fw)]).transposed()
    o.matrix_world = Matrix.Translation(Vector(pos)) @ R.to_4x4()
    bpy.context.scene.camera = o
    return o


def lights(objs, key=1.0, backdrop=False):
    for o in [o for o in bpy.data.objects if o.type == "LIGHT" or o.name == "Backdrop"]:
        bpy.data.objects.remove(o)
    return scene.studio(objs, backdrop=backdrop, key=key)


def on_white(src, dst):
    im = Image.open(src).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    bg.convert("RGB").save(dst)
    return dst


def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "molle_pouch.blend"))
    B, F = bpy.data.objects["SM_GPPouch"], bpy.data.objects["SM_GPPouch_Flap"]
    for o in bpy.data.objects:
        if o.type == "MESH" and o not in (B, F):
            o.hide_render = True
    os.makedirs(REND, exist_ok=True); os.makedirs(CMP, exist_ok=True)
    V, T = mesh_world([B, F])
    V *= 1.0
    c = (V.min(0) + V.max(0)) / 2
    s = bpy.context.scene
    s.view_settings.view_transform = "AgX"
    out = {}
    if "refs" in ONLY:
        res = {}
        for name in REFS:
            x, (W0, H0), iou = solve(name, V, T, c)
            k = 1024.0 / max(W0, H0)
            W, H = int(round(W0 * k)), int(round(H0 * k))
            lights([B, F], backdrop=False)
            bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
            bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
            make_cam(x, c, "REF_" + name, None)
            raw = os.path.join(CMP, f"raw_{name}.png")
            scene.render(raw, W, SMP, transparent=True, res_y=H)
            p = on_white(raw, os.path.join(REND, f"molle_pouch_ref_{name}.png"))
            out[name] = p
            res[name] = {"iou_solve": round(float(iou), 3), "camera": [round(float(v), 4) for v in x]}
            subprocess.run([sys.executable, os.path.join(ROOT, "tools/ref_compare.py"), os.path.join(REFD, REFS[name][0]), raw,
                            os.path.join(CMP, name), "--min-iou", "0"], check=False)
        json.dump(res, open(os.path.join(CMP, "cams.json"), "w"), indent=1)
    if "hero" in ONLY:
        lights([B, F], backdrop=True)
        scene.frame([B, F], yaw=32, pitch=20, lens=85, fill=0.82, name="Hero")
        out["hero"] = scene.render(os.path.join(REND, "molle_pouch.png"), 1024, SMP)
    if "open" in ONLY:
        F.rotation_euler.x = math.radians(100)
        lights([B, F], backdrop=True)
        scene.frame([B, F], yaw=28, pitch=30, lens=70, fill=0.8, name="Open")
        out["open"] = scene.render(os.path.join(REND, "molle_pouch_open.png"), 1024, SMP)
        F.rotation_euler.x = 0
        # parts: flap lifted off the body along its hinge normal
        F.location.y -= 0.06; F.location.z += 0.05; F.rotation_euler.x = math.radians(25)
        lights([B, F], backdrop=True)
        scene.frame([B, F], yaw=60, pitch=18, lens=70, fill=0.85, name="Parts")
        out["parts"] = scene.render(os.path.join(REND, "molle_pouch_parts.png"), 1024, SMP)
        F.location.y += 0.06; F.location.z -= 0.05; F.rotation_euler.x = 0
    if "closeup" in ONLY:
        # front webbing (row 3, the +x channels) and the top-right corner seam / zipper top, ~5.5 cm across
        lights([B, F], backdrop=True)
        tgt = Vector((0.050, -0.040, 0.128))
        cd = bpy.data.cameras.new("Close"); cd.lens = 100; cd.clip_start = 0.001
        cam = bpy.data.objects.new("Close", cd); bpy.context.scene.collection.objects.link(cam)
        cam.location = tgt + Vector((0.075, -0.17, 0.085))
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.scene.camera = cam
        out["closeup"] = scene.render(os.path.join(REND, "molle_pouch_closeup.png"), 1024, SMP)
    if "variants" in ONLY:
        names = [m.name for m in bpy.data.materials if m.name.startswith("MI_GPPouch_")]
        order = ["MI_GPPouch_RangerGreen", "MI_GPPouch_Coyote", "MI_GPPouch_Black", "MI_GPPouch_WolfGrey", "MI_GPPouch_MultiTerrain"]
        dups = []
        for i, nm in enumerate([n for n in order if n in names]):
            for o in (B, F):
                d = o.copy(); d.data = o.data.copy(); d.data.materials[0] = bpy.data.materials[nm]
                d.location.x += (i - 2) * 0.19
                d.rotation_euler.z = math.radians(-18)
                bpy.context.scene.collection.objects.link(d); d.hide_render = False
                dups.append(d)
        B.hide_render = F.hide_render = True
        lights(dups, backdrop=True)
        scene.frame(dups, yaw=0, pitch=12, lens=70, fill=0.92, name="Variants", aspect=0.42)
        out["variants"] = scene.render(os.path.join(REND, "molle_pouch_variants.png"), 1024, SMP, res_y=430)
        for d in dups:
            bpy.data.objects.remove(d)
        B.hide_render = F.hide_render = False
    if "sheet" in ONLY:
        names = ["molle_pouch.png", "molle_pouch_closeup.png", "molle_pouch_ref_front.png", "molle_pouch_ref_front34.png",
                 "molle_pouch_ref_back.png", "molle_pouch_ref_side.png", "molle_pouch_open.png", "molle_pouch_parts.png",
                 "molle_pouch_variants.png"]
        ims = [os.path.join(REND, n) for n in names if os.path.exists(os.path.join(REND, n))]
        subprocess.run([sys.executable, os.path.join(ROOT, "tools/sheet.py"), os.path.join(REND, "molle_pouch_sheet.jpg"),
                        "GPPouch - 6x6 utility pouch (LOD0 game mesh, M_Lab_Prop preview)"] + ims, check=False)
    print("RENDERS", json.dumps(out))


if __name__ == "__main__":
    main()
