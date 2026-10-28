"""bl_final.py - AK-74M materials, LODs, collision, sockets, mechanism, export and review renders.
usage: .venv/bin/python guns/ak74/bl_final.py [stage=all|export|render|exploded|fieldstrip] [samples=64]
in:  work/ak_low.blend (bl_low), work/textures.json (bl_bake)
out: fbx/ (SM_AK74_<Part> + UCX inside, fbx/lod/ _LOD1/_LOD2, SM_AK74 = merged ItemMesh without the magazine), sockets/,
     manifest.json, review/ (model-only renders: hero, sides, stock folded, exploded with callouts, field strip), ak74.blend
Frame (same as the SCAR): UE cm = G mm / 10, G = B - ORIGIN_B (origin at the grip top), +Y muzzle, +X gun-left, +Z up."""
import bpy, bmesh, os, sys, math, json
import numpy as np
from mathutils import Vector, Matrix
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import scene, master, export
import ak_spec as P
import ak_ref as R

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
STAGE = ARGS.get("stage", "all")
SMP = min(64, int(ARGS.get("samples", 64) or 64))
FBXD, REV = os.path.join(HERE, "fbx"), os.path.join(HERE, "review")
PARTS = ["Receiver", "DustCover", "Recoil", "Carrier", "Barrel", "Muzzle", "GasTube", "LowerHG", "Rod", "Selector",
         "Trigger", "Grip", "Mag", "Stock"]
FURNITURE = ("GasTube", "LowerHG", "Grip", "Mag", "Stock")
# variant parameters (linear). Black = the reference rifle: satin-black phosphate steel, matte black polymer a touch lighter
# (look_match.py refines these against the photos); Plum = the classic AK-74M plum furniture of the anatomy chart.
VARIANTS = {
    "Black": dict(ColorA=(0.026, 0.026, 0.028), RoughA=0.48, MetalA=0.35, ColorB=(0.034, 0.034, 0.035), RoughB=0.62, MetalB=0.0,
                  LabelRough=0.55, WearAmount=0.30, Dirt=0.30),
    "Plum": dict(ColorA=(0.026, 0.026, 0.028), RoughA=0.48, MetalA=0.35, ColorB=(0.075, 0.028, 0.045), RoughB=0.6, MetalB=0.0,
                 LabelRough=0.55, WearAmount=0.30, Dirt=0.30),
}
LOOK = os.path.join(HERE, "work", "look_fit.json")
if os.path.exists(LOOK):                                      # look_match.py fit (steel + polymer albedo, roughness)
    _l = json.load(open(LOOK))
    VARIANTS["Black"].update({k: (tuple(v) if isinstance(v, list) else v) for k, v in _l.get("variant", {}).items()})
LOD_PARTS = {"Receiver", "DustCover", "Carrier", "Barrel", "Muzzle", "GasTube", "LowerHG", "Grip", "Mag", "Stock", "Recoil"}


def b2bl(x, y, z):
    o = P.ORIGIN_B
    return Vector(((x - o[0]) * 0.001, -(y - o[1]) * 0.001, (z - o[2]) * 0.001))


def b2g(x, y, z):
    o = P.ORIGIN_B
    return (x - o[0], y - o[1], z - o[2])


yb = lambda x: R.bp(x, 285.98 + R.BORE_SLOPE * (x - 1030.0))[0]
# sockets in the B frame (mm) + rotation (pitch, yaw, roll deg)
SOCKETS_B = {
    "Muzzle": ((0, P.MUZZLE_Y, 0), (0, 0, 0)),
    "FrontSight": ((0, *P.FRONT_SIGHT_TOP), (0, 0, 0)),
    "RearSight": ((0, *P.REAR_NOTCH), (0, 0, 0)),
    "Aim": ((0, P.REAR_NOTCH[0], P.REAR_NOTCH[1] - 1.0), (0, 0, 0)),
    "Optic_Socket": ((P.HW_REC + 5.0, *R.lp2b(792.0, 289.0)), (0, 0, 90)),
    "EjectionPort": ((-P.HW_REC, R.bp(650.0, 267.5)[0], R.bp(650.0, 267.5)[1]), (0, 0, -90)),
    "MagazineSocket": ((0, *P.MAG_TOP), (0, 0, 0)),
    "Chamber": ((0, yb(690.0), 0), (0, 0, 0)),
    "HandPose_R": ((0, R.bp(485.0, 330.0)[0], R.bp(485.0, 330.0)[1]), (0, 0, 0)),
    "HandPose_L": ((0, R.bp(800.0, 300.0)[0], R.bp(800.0, 300.0)[1]), (0, 0, 0)),
    "Stock": ((0, P.BUTT_Y, R.bp(200.0, 336.0)[1]), (0, 0, 0)),
    "Pivot_StockHinge": ((P.HINGE[0], P.HINGE[1], 0.0), (0, 0, 0)),
    "Pivot_Magazine": ((0, *P.MAG_PIVOT), (0, 0, 0)),
    "Pivot_DustCover": ((0, *P.DC_PIVOT), (0, 0, 0)),
    "Pivot_Selector": ((-P.HW_REC, *P.SELECTOR_AXIS), (0, 0, 0)),
    "Pivot_Trigger": ((0, *R.bp(543.5, 311.5)), (0, 0, 0)),
    "ChargingHandle": ((-P.HW_REC - 13.0, *P.CH_POS), (0, 0, -90)),
}
MECH = {
    "SM_AK74_Carrier": dict(type="slide", axis_b=[0, -1, 0], travel_cm=9.6, pivot="assembled pose",
                            note="bolt carrier + piston + charging handle; the bolt (same mesh) rides with it. Travel back = "
                                 "the charging handle from the port front to the rear of the ejection port"),
    "SM_AK74_Recoil": dict(type="compress", axis_b=[0, 1, 0], note="spring compresses with the carrier; button stays"),
    "SM_AK74_DustCover": dict(type="hinge+slide", pivot="Pivot_DustCover", axis_b=[1, 0, 0], angle_deg=-14.0, then_slide_cm=-3.0,
                              note="field strip: press the guide button, lift the cover at the front about its rear, slide off"),
    "SM_AK74_Selector": dict(type="hinge", pivot="Pivot_Selector", axis_b=[1, 0, 0],
                             positions_deg={"safe": 0.0, "auto": -17.0, "semi": -33.0}, note="front end moves down"),
    "SM_AK74_Trigger": dict(type="hinge", pivot="Pivot_Trigger", axis_b=[1, 0, 0], pull_deg=13.0),
    "SM_AK74_Mag": dict(type="rock+drop", socket="MagazineSocket", pivot="Pivot_Magazine", axis_b=[1, 0, 0], rock_deg=14.0,
                        note="AK insertion: hook the front lug, rock the rear up until the catch clicks (reverse to remove)"),
    "SM_AK74_Stock": dict(type="hinge", pivot="Pivot_StockHinge", axis_b=[0, 0, 1], angle_deg=180.0,
                          note="folds to the LEFT along the receiver (positive rotation about +Z through the hinge)"),
    "SM_AK74_GasTube": dict(type="field-strip", note="gas tube + upper handguard lift off after the RSB lever is rotated up"),
}


def materials():
    T = json.load(open(os.path.join(HERE, "work", "textures.json")))
    mats = {}
    for n in PARTS:
        t = {k: T[n][k] for k in ("BaseColor", "Normal", "ORM", "Mask")}
        for v, prm in VARIANTS.items():
            m = master.material(f"MI_AK74_{n}_{v}", t, prm, label=T[n].get("Label"))
            m.use_fake_user = True
            mats[(n, v)] = m
    return mats


def assign(objs, mats, v):
    for n, o in objs.items():
        o.data.materials.clear(); o.data.materials.append(mats[(n, v)])


def hulls(objs):
    out = {}
    for n, o in objs.items():
        if n in ("Recoil", "Rod", "Trigger", "Selector", "Carrier"):
            out[n] = []
            continue
        h = export.hull(o, f"UCX_SM_AK74_{n}_00", 48); h.hide_render = True
        out[n] = [h]
    return out


def lods(objs):
    out = {}
    for n, o in objs.items():
        if n in LOD_PARTS:
            out[n] = [export.lod(o, r, f"SM_AK74_{n}_LOD{i + 1}") for i, r in enumerate((0.45, 0.2))]
            for l in out[n]:
                l.hide_render = True; l.hide_set(True)
    return out


def sockets_json():
    out = {}
    for k, ((x, y, z), rot) in SOCKETS_B.items():
        gx, gy, gz = b2g(x, y, z)
        out[k] = dict(location_cm=[round(gx / 10, 3), round(gy / 10, 3), round(gz / 10, 3)], rotation_pyr=list(rot), mesh="SM_AK74")
    return out


def export_all(objs, mats):
    assign(objs, mats, "Black")
    H, L = hulls(objs), lods(objs)
    tri = {}
    for n, o in objs.items():
        tri[o.name] = scene.tris(o)
        export.fbx([o] + H[n], f"{FBXD}/{o.name}.fbx")
        for l in L.get(n, []):
            tri[l.name] = scene.tris(l); export.fbx([l], f"{FBXD}/lod/{l.name}.fbx")
    parts = [objs[n] for n in PARTS if n != "Mag"]
    cps = []
    for o in parts:
        c = o.copy(); c.data = o.data.copy(); bpy.context.scene.collection.objects.link(c); cps.append(c)
    with bpy.context.temp_override(active_object=cps[0], selected_editable_objects=cps, object=cps[0]):
        bpy.ops.object.join()
    full = cps[0]; full.name = full.data.name = "SM_AK74"
    ucx = [h.copy() for hs in H.values() for h in hs if "Mag" not in h.name]
    for i, h in enumerate(ucx):
        h.data = h.data.copy(); h.name = f"UCX_SM_AK74_{i:02d}"; bpy.context.scene.collection.objects.link(h)
    export.fbx([full] + ucx, f"{FBXD}/SM_AK74.fbx")
    tri["SM_AK74"] = scene.tris(full)
    full_l = [export.lod(full, r, f"SM_AK74_LOD{i + 1}") for i, r in enumerate((0.45, 0.2))]
    for l in full_l:
        tri[l.name] = scene.tris(l); export.fbx([l], f"{FBXD}/lod/{l.name}.fbx")
    for x in [full] + ucx + full_l:
        x.hide_render = True; x.hide_set(True)
    os.makedirs(os.path.join(HERE, "sockets"), exist_ok=True)
    json.dump(sockets_json(), open(os.path.join(HERE, "sockets", "SM_AK74_sockets.json"), "w"), indent=1)
    json.dump(MECH, open(os.path.join(HERE, "sockets", "SM_AK74_mechanism.json"), "w"), indent=1)
    T = json.load(open(os.path.join(HERE, "work", "textures.json")))
    rel = lambda p: os.path.relpath(p, HERE)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "AK74M", "variant": "AK-74M 5.45x39, 415 mm barrel, black polymer furniture, side-folding stock (Black / Plum)",
        "category": "weapon", "frame": "UE cm = G mm / 10; G = B - ORIGIN_B (origin at the grip top); +Y muzzle, +X gun-left, +Z up",
        "meshes": {o.name: {"tex": {k: rel(v) for k, v in T[n].items()}, "lods": [l.name for l in L.get(n, [])], "ucx": [h.name for h in H[n]]}
                   for n, o in objs.items()},
        "item_mesh": "SM_AK74 (merged, no magazine)", "tris": tri,
        "lod0_total": sum(v for k, v in tri.items() if k.startswith("SM_AK74_") and "_LOD" not in k),
        "variants": {v: {n: f"MI_AK74_{n}_{v}" for n in PARTS} for v in VARIANTS}, "variant_params": VARIANTS,
        "sockets": "sockets/SM_AK74_sockets.json", "mechanism": "sockets/SM_AK74_mechanism.json",
        "textures": {"normal": "OpenGL (flip green in UE)", "ORM": "linear", "Mask": "linear RGBA (R steel finish A, G polymer B, B print, A wear)",
                     "Label": "sRGB RGBA print (sight numbers, stamped serial)"}})
    print("TRIS", tri)
    return tri


def stock_fold(o, deg=180.0):
    """rotate the Stock object about the vertical hinge axis (pivot in the game frame)"""
    p = b2bl(P.HINGE[0], P.HINGE[1], 0.0)
    M = Matrix.Translation(p) @ Matrix.Rotation(math.radians(deg), 4, "Z") @ Matrix.Translation(-p)
    o.matrix_world = M @ o.matrix_world


EXPLODE = {"Receiver": (0, 0, 0), "DustCover": (0, 0.0, 0.13), "Recoil": (0, 0.0, 0.22), "Carrier": (0, -0.06, 0.32),
           "Barrel": (0, -0.14, 0.0), "Muzzle": (0, -0.24, 0.0), "GasTube": (0, -0.08, 0.14), "LowerHG": (0, -0.08, -0.12),
           "Rod": (0, -0.12, -0.20), "Selector": (-0.09, 0.0, 0.0), "Trigger": (0, 0.0, -0.10), "Grip": (0, 0.05, -0.22),
           "Mag": (0, -0.06, -0.33), "Stock": (0, 0.16, 0.0)}
LABELS = {"Receiver": "Stamped receiver, rear sight, side rail, trigger guard, mag catch", "DustCover": "Dust cover (stamped)",
          "Recoil": "Recoil spring + guide", "Carrier": "Bolt carrier + piston + charging handle, bolt", "Barrel": "Barrel, gas block, front sight",
          "Muzzle": "AK-74 muzzle brake", "GasTube": "Gas tube + upper handguard", "LowerHG": "Lower handguard",
          "Rod": "Cleaning rod", "Selector": "Safety / selector", "Trigger": "Trigger", "Grip": "Pistol grip (checkered)",
          "Mag": "30-round polymer magazine", "Stock": "Side-folding polymer stock"}


def annotate(path, cam, objs, title):
    from PIL import Image, ImageDraw, ImageFont
    from bpy_extras.object_utils import world_to_camera_view
    im = Image.open(path).convert("RGB"); W, Hh = im.size; pad = 54
    can = Image.new("RGB", (W, Hh + pad), (22, 23, 26)); can.paste(im, (0, pad)); d = ImageDraw.Draw(can)
    fdir = os.path.join(HERE, "../../fonts")
    fb = ImageFont.truetype(os.path.join(fdir, "WorkSans-Bold.ttf"), 30); fs = ImageFont.truetype(os.path.join(fdir, "WorkSans-Bold.ttf"), 19)
    d.text((16, 10), title, font=fb, fill=(235, 235, 235))
    placed = []
    for n, o in objs.items():
        c = sum((o.matrix_world @ Vector(b) for b in o.bound_box), Vector()) / 8
        p = world_to_camera_view(bpy.context.scene, cam, c)
        x, y = p.x * W, (1 - p.y) * Hh + pad
        lx = x + (60 if x < W * 0.5 else -60); ly = y - 46 if p.y > 0.5 else y + 40
        while any(abs(ly - q) < 26 and abs(lx - r) < 320 for r, q in placed):
            ly += 28
        placed.append((lx, ly))
        txt = LABELS[n]; tw = d.textlength(txt, font=fs); tx = lx if x < W * 0.5 else lx - tw
        d.line([(x, y), (lx, ly + 11)], fill=(255, 196, 64), width=2)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 196, 64))
        d.rectangle([tx - 5, ly - 3, tx + tw + 5, ly + 25], fill=(20, 20, 22))
        d.text((tx, ly), txt, font=fs, fill=(240, 240, 240))
    can.save(path.replace(".png", ".jpg"), quality=92)


def clear_scene_extras(objs):
    keep = {o.name for o in objs.values()}
    for o in list(bpy.data.objects):
        if o.type in ("LIGHT", "CAMERA") or (o.type == "MESH" and o.name not in keep and not o.hide_render):
            bpy.data.objects.remove(o)


def renders(objs, mats):
    os.makedirs(REV, exist_ok=True)
    allo = list(objs.values())
    scene.studio(allo)
    bpy.context.scene.cycles.max_bounces = 6
    for v in VARIANTS:
        assign(objs, mats, v)
        scene.frame(allo, yaw=-62, pitch=12, lens=70, fill=0.9, name="hero_" + v, aspect=0.56)
        scene.render(f"{REV}/hero_{v.lower()}.png", 1600, SMP, res_y=900)
    assign(objs, mats, "Black")
    scene.frame(allo, yaw=-90, pitch=0, lens=135, fill=0.95, name="side_r", aspect=0.42)
    scene.render(f"{REV}/side_right.png", 1800, SMP, res_y=756)
    scene.frame(allo, yaw=90, pitch=0, lens=135, fill=0.95, name="side_l", aspect=0.42)
    scene.render(f"{REV}/side_left.png", 1800, SMP, res_y=756)
    stock_fold(objs["Stock"])
    scene.frame(allo, yaw=128, pitch=26, lens=70, fill=0.9, name="folded", aspect=0.6)
    scene.render(f"{REV}/folded.png", 1600, SMP, res_y=960)
    stock_fold(objs["Stock"], -180.0)
    for nm, tgt, yaw, pitch, lens in (("close_receiver", ["Receiver", "DustCover", "Selector"], -75, 10, 60),
                                      ("close_front", ["Muzzle", "Barrel"], -125, 14, 70),
                                      ("close_grip", ["Grip", "Trigger"], -60, 8, 70)):
        scene.frame([objs[t] for t in tgt], yaw=yaw, pitch=pitch, lens=lens, fill=0.75, name=nm, aspect=0.75)
        scene.render(f"{REV}/{nm}.png", 1200, SMP, res_y=900)


def exploded(objs, mats):
    clear_scene_extras(objs)
    assign(objs, mats, "Black")
    for n, o in objs.items():
        o.location = EXPLODE[n]
    bpy.context.view_layer.update()
    allo = list(objs.values())
    scene.studio(allo)
    cam = scene.frame(allo, yaw=-90, pitch=6, lens=90, fill=0.86, name="expl", aspect=0.62)
    scene.render(f"{REV}/exploded.png", 2000, SMP, res_y=1240)
    annotate(f"{REV}/exploded.png", cam, objs, f"AK-74M - exploded ({len(objs)} game meshes)")
    for o in objs.values():
        o.location = (0, 0, 0)


# field strip, top-down on a neutral mat: parts lying on their left sides, the standard AK field-strip order
FIELD = {"Mag": (-0.10, 0.30), "Rod": (0.0, 0.22), "DustCover": (-0.04, 0.13), "Recoil": (-0.04, 0.06), "Carrier": (0.04, 0.0),
         "GasTube": (0.10, 0.08)}


def fieldstrip(objs, mats):
    clear_scene_extras(objs)
    assign(objs, mats, "Black")
    main = ["Receiver", "Barrel", "Muzzle", "LowerHG", "Selector", "Trigger", "Grip", "Stock"]
    root = bpy.data.objects.new("G_main", None); bpy.context.scene.collection.objects.link(root)
    for n in main:
        objs[n].parent = root
    root.rotation_euler = (0, math.radians(90), 0)                     # lying on its left side, right side up
    root.location = (0, 0, 0)
    bpy.context.view_layer.update()
    lo, hi = scene.bounds([objs[n] for n in main]); root.location.z -= lo.z
    root.location.x -= (lo.x + hi.x) / 2 + 0.16
    for n, (dx, dy) in FIELD.items():
        o = objs[n]
        o.rotation_euler = (0, math.radians(90), 0)
        bpy.context.view_layer.update()
        a, b = scene.bounds([o])
        o.location = (o.location.x - (a.x + b.x) / 2 + dx + 0.06, o.location.y - (a.y + b.y) / 2 + dy - 0.05, o.location.z - a.z)
    bpy.context.view_layer.update()
    allo = list(objs.values())
    scene.studio(allo)
    cam = scene.frame(allo, yaw=-90, pitch=89, lens=50, fill=0.9, name="fs", aspect=0.7)
    scene.render(f"{REV}/fieldstrip.png", 1800, SMP, res_y=1260)


def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "work", "ak_low.blend"))
    hc = bpy.data.collections.get("HIGH")
    if hc:
        for o in list(hc.objects):
            bpy.data.objects.remove(o)
    objs = {n: bpy.data.objects["SM_AK74_" + n] for n in PARTS}
    mats = materials()
    if STAGE in ("all", "export"):
        export_all(objs, mats)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "ak74.blend"), relative_remap=True)
    if STAGE in ("all", "render"):
        renders(objs, mats)
    if STAGE in ("all", "exploded"):
        exploded(objs, mats)
    if STAGE in ("all", "fieldstrip"):
        bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "ak74.blend"))
        objs = {n: bpy.data.objects["SM_AK74_" + n] for n in PARTS}
        fieldstrip(objs, materials())


if __name__ == "__main__":
    main()
