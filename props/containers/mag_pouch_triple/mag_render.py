"""Renders for MagPouch3: hero (closed), flaps open, loaded (magazines), reference-angle matches (front closed / open 3/4 /
back), close-up, skins, contact sheet.  .venv/bin/python props/containers/mag_pouch_triple/mag_render.py [--fast] [--only ...]
Cameras for the reference angles are solved on the silhouette (see ../molle_pouch/pouch_render.py); the photo comparison
sheets go to work/mag_pouch_triple/compare/ (not committed), never into renders/."""
import os, sys, math, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../molle_pouch"))
import bpy
from mathutils import Vector
from lab import scene
import pouch_render as PR

FAST = "--fast" in sys.argv
SMP = 16 if FAST else 64
REND = os.path.join(HERE, "renders")
CMP = os.path.join(ROOT, "work/mag_pouch_triple/compare")
PR.REFD = os.path.join(HERE, "refs_user")
PR.REFS = {"front_closed": ("front_closed.jpg", 0, 4), "open34": ("open34.jpg", 28, 10), "back": ("back.jpg", 180, 4)}
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else ["hero", "refs", "open", "closeup", "variants", "sheet"]
OPEN = math.radians(-115)


def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "mag_pouch_triple.blend"))
    B = bpy.data.objects["SM_MagPouch3"]
    flaps = [bpy.data.objects[f"SM_MagPouch3_Flap{i}"] for i in (1, 2, 3)]
    M = bpy.data.objects["SM_MagPouch3_Mags"]
    for o in bpy.data.objects:
        if o.type == "MESH" and o not in [B, M] + flaps:
            o.hide_render = True
    os.makedirs(REND, exist_ok=True); os.makedirs(CMP, exist_ok=True)
    s = bpy.context.scene
    out = {}

    def state(open_, mags):
        for f in flaps:
            f.rotation_euler.x = OPEN if open_ else 0.0
        M.hide_render = not mags
        bpy.context.view_layer.update()
    if "refs" in ONLY:
        res = {}
        for name in PR.REFS:
            open_ = name == "open34"
            state(open_, True)
            solve_objs = [B, M] if open_ else [B] + flaps
            V, T = PR.mesh_world(solve_objs)
            c = (V.min(0) + V.max(0)) / 2
            x, (W0, H0), iou = PR.solve(name, V, T, c)
            k = 1024.0 / max(W0, H0)
            W, H = int(round(W0 * k)), int(round(H0 * k))
            vis = [B, M] + flaps
            PR.lights(vis, backdrop=False)
            s.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
            s.world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
            PR.make_cam(x, c, "REF_" + name, None)
            raw = os.path.join(CMP, f"raw_{name}.png")
            scene.render(raw, W, SMP, transparent=True, res_y=H)
            out[name] = PR.on_white(raw, os.path.join(REND, f"mag_pouch_triple_ref_{name}.png"))
            res[name] = {"iou_solve": round(float(iou), 3), "camera": [round(float(v), 4) for v in x]}
            subprocess.run([sys.executable, os.path.join(ROOT, "tools/ref_compare.py"), os.path.join(PR.REFD, PR.REFS[name][0]), raw,
                            os.path.join(CMP, name), "--min-iou", "0"], check=False)
        json.dump(res, open(os.path.join(CMP, "cams.json"), "w"), indent=1)
    allv = [B, M] + flaps
    if "hero" in ONLY:
        state(False, True)
        PR.lights(allv, backdrop=True)
        scene.frame(allv, yaw=30, pitch=18, lens=85, fill=0.84, name="Hero")
        out["hero"] = scene.render(os.path.join(REND, "mag_pouch_triple.png"), 1024, SMP)
    if "open" in ONLY:
        state(True, True)
        PR.lights(allv, backdrop=True)
        scene.frame(allv, yaw=28, pitch=24, lens=85, fill=0.84, name="Open")
        out["open"] = scene.render(os.path.join(REND, "mag_pouch_triple_open.png"), 1024, SMP)
        state(True, False)
        PR.lights([B] + flaps, backdrop=True)
        scene.frame([B] + flaps, yaw=20, pitch=30, lens=85, fill=0.84, name="OpenEmpty")
        out["open_empty"] = scene.render(os.path.join(REND, "mag_pouch_triple_open_empty.png"), 1024, SMP)
        state(False, True)
    if "closeup" in ONLY:
        state(False, True)
        PR.lights(allv, backdrop=True)
        tgt = Vector((0.095, -0.04, 0.075))
        cd = bpy.data.cameras.new("Close"); cd.lens = 100; cd.clip_start = 0.001
        cam = bpy.data.objects.new("Close", cd); s.collection.objects.link(cam)
        cam.location = tgt + Vector((0.07, -0.19, 0.06))
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        s.camera = cam
        out["closeup"] = scene.render(os.path.join(REND, "mag_pouch_triple_closeup.png"), 1024, SMP)
    if "variants" in ONLY:
        state(False, True)
        order = ["Navy", "RangerGreen", "Coyote", "Black", "MultiTerrain"]
        dups = []
        for i, nm in enumerate(order):
            m = bpy.data.materials.get(f"MI_MagPouch3_{nm}")
            if not m:
                continue
            for o in allv:
                d = o.copy(); d.data = o.data.copy(); d.data.materials[0] = m
                d.location.x += (i - 2) * 0.31
                s.collection.objects.link(d); d.hide_render = False
                dups.append(d)
        for o in allv:
            o.hide_render = True
        PR.lights(dups, backdrop=True)
        scene.frame(dups, yaw=0, pitch=10, lens=70, fill=0.94, name="Variants", aspect=0.3)
        out["variants"] = scene.render(os.path.join(REND, "mag_pouch_triple_variants.png"), 1024, SMP, res_y=310)
    if "sheet" in ONLY:
        names = ["mag_pouch_triple.png", "mag_pouch_triple_open.png", "mag_pouch_triple_ref_front_closed.png",
                 "mag_pouch_triple_ref_open34.png", "mag_pouch_triple_ref_back.png", "mag_pouch_triple_closeup.png",
                 "mag_pouch_triple_open_empty.png", "mag_pouch_triple_variants.png"]
        ims = [os.path.join(REND, n) for n in names if os.path.exists(os.path.join(REND, n))]
        subprocess.run([sys.executable, os.path.join(ROOT, "tools/sheet.py"), os.path.join(REND, "mag_pouch_triple_sheet.jpg"),
                        "MagPouch3 - triple AR mag pouch with covers (LOD0 game mesh, M_Lab_Prop preview)"] + ims, check=False)
    print("RENDERS", json.dumps(out))


if __name__ == "__main__":
    main()
