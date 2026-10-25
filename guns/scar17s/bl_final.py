"""bl_final.py - SCAR 17S materials, LODs, collision, sockets, export and review renders.
usage: .venv/bin/python guns/scar17s/bl_final.py [stage=all|export|render] [samples=64]
in:  work/scar_low.blend (bl_low), work/textures.json (bl_bake)
out: fbx/ (SM_SCAR17S_<Part> + _LOD1/_LOD2 + UCX inside, SM_SCAR17S = merged ItemMesh), sockets/, manifest.json,
     review/ (hero, side, exploded with callouts, variants, closeups), scar17s.blend"""
import bpy, bmesh, os, sys, math, json
import numpy as np
from mathutils import Vector
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import scene, master, export
import scar_spec as P

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
STAGE = ARGS.get("stage", "all")
SMP = int(ARGS.get("samples", 64) or 64)
FBXD, REV = os.path.join(HERE, "fbx"), os.path.join(HERE, "review")
PARTS = ["Upper", "Barrel", "Muzzle", "Bolt", "Lower", "Grip", "VGrip", "Stock", "Mag", "Round"]
RECEIVER = ("Upper", "Lower", "Grip", "VGrip", "Stock", "Mag")
# FDE measured on the profile master (photo ratios: anodised R/G 1.40 B/G 0.58; polymer R/G 1.7 B/G 0.43; grips paler)
VARIANTS = {
    "FDE": dict(ColorA=(0.1440, 0.0868, 0.0288), RoughA=0.5, MetalA=0.0, ColorB=(0.3639, 0.1948, 0.0458), RoughB=0.62, MetalB=0.0,
                LabelRough=0.6, WearAmount=0.12, Dirt=0.15),                    # look_match.py fit to the hero34 photo
    "Black": dict(ColorA=(0.030, 0.030, 0.032), RoughA=0.46, MetalA=0.0, ColorB=(0.032, 0.031, 0.030), RoughB=0.62, MetalB=0.0,
                  LabelRough=0.6, WearAmount=0.42, Dirt=0.35),
    "Camo": dict(ColorA=(0.34, 0.243, 0.141), RoughA=0.5, MetalA=0.0, ColorB=(0.32, 0.192, 0.083), RoughB=0.64, MetalB=0.0,
                 LabelRough=0.62, WearAmount=0.3, Dirt=0.45),
}
GRIP_FDE = dict(ColorB=(0.3328, 0.1926, 0.0619))                    # grips: their own fit (look_match.py)
NEUTRAL = dict(ColorA=(0.5, 0.5, 0.5), RoughA=0.4, MetalA=0.0, ColorB=(0.5, 0.5, 0.5), RoughB=0.4, MetalB=0.0,
               LabelRough=0.5, WearAmount=0.55, Dirt=0.4)
LOD = {"Upper", "Bolt", "Lower", "Grip", "VGrip", "Stock", "Mag", "Muzzle"}


def g2bl(y, z, x=0.0):
    """G frame mm (origin = grip top) -> Blender metres"""
    return Vector((x * 0.001, -y * 0.001, z * 0.001))


def b2g(x, y, z):
    o = P.ORIGIN_B
    return (x - o[0], y - o[1], z - o[2])


# sockets in the B frame (mm); converted to UE cm (= G mm / 10) with pitch/yaw/roll in degrees
SOCKETS_B = {
    "Muzzle": ((0, P.MUZZLE_Y, 0), (0, 0, 0)),
    "Aim": ((0, P.REAR_APERTURE[0], P.REAR_APERTURE[1]), (0, 0, 0)),     # deployed rear aperture centre
    "FrontSight": ((0, P.FRONT_SIGHT_TOP[0], P.FRONT_SIGHT_TOP[1]), (0, 0, 0)),
    "Optic_Socket": ((0, -60.0, P.RAIL_TOP), (0, 0, 0)),
    "Grip_Socket": ((0, P.VGRIP[0], P.FE_BOT), (0, 0, 0)),             # bottom M-LOK, where the vertical grip sits
    "TacticalSocket": ((-27.8 * 0.75, 240.0, P.SIDE_RAIL_Z), (0, 0, -90)),     # right side M-LOK
    "MagazineSocket": ((0, (P.MAG_Y[0] + P.MAG_Y[1]) / 2, P.MAG_TOP), (0, 0, 0)),
    "Chamber": ((0, 20.0, 0), (0, 0, 0)),
    "EjectionPort": ((-27.8 * 0.75, -25.0, 7.0), (0, 0, -90)),
    "HandPose_R": ((0, P.GRIP_TOP[0] - 10, P.GRIP_TOP[1] - 45), (0, 0, 0)),
    "HandPose_L": ((0, P.VGRIP[0], P.FE_BOT - 45), (0, 0, 0)),
    "Stock": ((0, P.BUTT_Y, -40.0), (0, 0, 0)),
    "StockHinge": ((P.HINGE[0], P.HINGE[1], 0.0), (0, 0, 0)),
}
MECH = {
    "SM_SCAR17S_Bolt": dict(axis_ue=(0, -1, 0), travel_cm=P.BOLT_TRAVEL / 10, pivot="assembled pose", linked_to=None),
    "SM_SCAR17S_Upper": dict(note="the NRCH charging handle (left) is part of the Upper mesh: it stays forward when firing"),
    "SM_SCAR17S_Stock": dict(axis_ue="vertical through StockHinge", angle_deg=-180, pivot="StockHinge",
                             note="folds to the gun's right; rotate about +Z (UE) at StockHinge"),
    "SM_SCAR17S_Mag": dict(socket="MagazineSocket", note="drops along -Z (UE)"),
}


def materials():
    T = json.load(open(os.path.join(HERE, "work", "textures.json")))
    mats = {}
    for n in PARTS:
        t = {k: T[n][k] for k in ("BaseColor", "Normal", "ORM", "Mask")}
        if n in RECEIVER:
            for v, prm in VARIANTS.items():
                lab = T[n]["Label_Camo"] if v == "Camo" else T[n]["Label_None"]
                prm = dict(prm, **GRIP_FDE) if (n in ("Grip", "VGrip") and v != "Black") else prm
                m = master.material(f"MI_SCAR17S_{n}_{v}", t, prm, label=lab)
                m.use_fake_user = True
                mats[(n, v)] = m
        else:
            m = master.material(f"MI_SCAR17S_{n}", t, NEUTRAL, label=None)
            m.use_fake_user = True
            for v in VARIANTS:
                mats[(n, v)] = m
    return mats


def assign(objs, mats, v):
    for n, o in objs.items():
        o.data.materials.clear()
        o.data.materials.append(mats[(n, v)])


def region_hull(o, name, sel):
    co = np.array([v.co[:] for v in o.data.vertices]); G = co * np.array([1000, -1000, 1000])
    pts = co[sel(G)]
    me = bpy.data.meshes.new(name + "_pts"); me.from_pydata([tuple(p) for p in pts], [], [])
    tmp = bpy.data.objects.new(name + "_pts", me); bpy.context.scene.collection.objects.link(tmp)
    h = export.hull(tmp, name, 64)
    bpy.data.objects.remove(tmp)
    h.hide_render = True
    return h


def hulls(objs):
    oz, oy = P.ORIGIN_B[2], P.ORIGIN_B[1]
    by = lambda G: G[:, 1] + oy                                           # G y -> B y
    U = {"SM_SCAR17S_Upper": [("00", lambda G: by(G) < 60), ("01", lambda G: by(G) >= 50)],
         "SM_SCAR17S_Lower": [("00", lambda G: by(G) > -110), ("01", lambda G: by(G) <= -100)],
         "SM_SCAR17S_Stock": [("00", lambda G: by(G) > -390), ("01", lambda G: by(G) <= -380)]}
    out = {}
    for n, o in objs.items():
        key = "SM_SCAR17S_" + n
        if key in U:
            out[n] = [region_hull(o, f"UCX_{key}_{k}", s) for k, s in U[key]]
        elif n != "Round":
            h = export.hull(o, f"UCX_{key}_00", 48); h.hide_render = True
            out[n] = [h]
        else:
            out[n] = []
    return out


def lods(objs):
    out = {}
    for n, o in objs.items():
        if n in LOD:
            rs = (0.45, 0.3) if n == "Bolt" else (0.45, 0.18)                # the bolt collapses at 18 %
            out[n] = [export.lod(o, r, f"SM_SCAR17S_{n}_LOD{i + 1}") for i, r in enumerate(rs)]
            for l in out[n]:
                l.hide_render = True; l.hide_set(True)
    return out


def sockets_json():
    out = {}
    for k, ((x, y, z), rot) in SOCKETS_B.items():
        gx, gy, gz = b2g(x, y, z)
        out[k] = dict(location_cm=[round(gx / 10, 3), round(gy / 10, 3), round(gz / 10, 3)], rotation_pyr=list(rot), mesh="SM_SCAR17S")
    return out


def export_all(objs, mats):
    assign(objs, mats, "FDE")
    H, L = hulls(objs), lods(objs)
    tri = {}
    for n, o in objs.items():
        tri[o.name] = scene.tris(o)
        export.fbx([o] + H[n], f"{FBXD}/{o.name}.fbx")
        for l in L.get(n, []):
            tri[l.name] = scene.tris(l)
            export.fbx([l], f"{FBXD}/lod/{l.name}.fbx")
    # merged ItemMesh (everything but the magazine and the loose round), one slot per part
    parts = [objs[n] for n in PARTS if n not in ("Mag", "Round")]
    cps = []
    for o in parts:
        c = o.copy(); c.data = o.data.copy(); bpy.context.scene.collection.objects.link(c); cps.append(c)
    with bpy.context.temp_override(active_object=cps[0], selected_editable_objects=cps, object=cps[0]):
        bpy.ops.object.join()
    full = cps[0]; full.name = full.data.name = "SM_SCAR17S"
    ucx = [h.copy() for hs in H.values() for h in hs if "Mag" not in h.name]
    for i, h in enumerate(ucx):
        h.data = h.data.copy(); h.name = f"UCX_SM_SCAR17S_{i:02d}"; bpy.context.scene.collection.objects.link(h)
    export.fbx([full] + ucx, f"{FBXD}/SM_SCAR17S.fbx")
    tri["SM_SCAR17S"] = scene.tris(full)
    full_l = [export.lod(full, r, f"SM_SCAR17S_LOD{i + 1}") for i, r in enumerate((0.45, 0.18))]
    for i, l in enumerate(full_l):
        tri[l.name] = scene.tris(l); export.fbx([l], f"{FBXD}/lod/{l.name}.fbx")
    for x in [full] + ucx + full_l:
        x.hide_render = True; x.hide_set(True)
    os.makedirs(os.path.join(HERE, "sockets"), exist_ok=True)
    json.dump(sockets_json(), open(os.path.join(HERE, "sockets", "SM_SCAR17S_sockets.json"), "w"), indent=1)
    json.dump(MECH, open(os.path.join(HERE, "sockets", "SM_SCAR17S_mechanism.json"), "w"), indent=1)
    T = json.load(open(os.path.join(HERE, "work", "textures.json")))
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "SCAR17S", "variant": "FN SCAR 17S NRCH, 16.25 in, 7.62x51 (FDE / Black / Camo skins)", "category": "weapon",
        "frame": "UE cm = G mm / 10; G = B - ORIGIN_B (origin at the grip top); +Y muzzle, +X gun-left, +Z up",
        "meshes": {o.name: {"tex": T[n], "lods": [l.name for l in L.get(n, [])], "ucx": [h.name for h in H[n]]} for n, o in objs.items()},
        "item_mesh": "SM_SCAR17S (merged, no magazine)", "tris": tri, "lod0_total": sum(v for k, v in tri.items() if k.startswith("SM_SCAR17S_") and "_LOD" not in k),
        "variants": {v: {n: f"MI_SCAR17S_{n}_{v}" for n in RECEIVER} for v in VARIANTS}, "variant_params": VARIANTS,
        "steel_params": NEUTRAL, "sockets": "sockets/SM_SCAR17S_sockets.json", "mechanism": "sockets/SM_SCAR17S_mechanism.json",
        "textures": {"normal": "OpenGL (flip green in UE)", "ORM": "linear", "Mask": "linear RGBA (R anodised, G polymer, B print, A wear)"}})
    print("TRIS", tri)
    return tri


EXPLODE = {"Upper": (0, 0, 0.0), "Barrel": (0, -0.22, 0.0), "Muzzle": (0, -0.32, 0.0), "Bolt": (0, 0.04, 0.15),
           "Lower": (0, 0.0, -0.13), "Grip": (0, 0.07, -0.27), "VGrip": (0, 0.0, -0.13), "Stock": (0, 0.16, 0.0),
           "Mag": (0, -0.06, -0.33), "Round": (0, -0.20, -0.33)}
LABELS = {"Upper": "Upper: 1913 top rail, M-LOK handguard, NRCH charging handle, QD cups, folding sights",
          "Barrel": "16.25 in barrel", "Muzzle": "Muzzle brake (2 side ports, top vents)", "Bolt": "Bolt carrier + 7-lug rotating bolt",
          "Lower": "Polymer lower: trigger, ambi selector + mag release, bolt catch, pins",
          "Grip": "Pistol grip (stipple panels)", "VGrip": "M-LOK vertical grip", "Stock": "Side-folding telescoping stock + buttpad",
          "Mag": "20-round magazine", "Round": "7.62x51 cartridge"}


def annotate(path, cam, objs, title):
    from PIL import Image, ImageDraw, ImageFont
    from bpy_extras.object_utils import world_to_camera_view
    im = Image.open(path).convert("RGB")
    W, Hh = im.size
    pad = 54
    can = Image.new("RGB", (W, Hh + pad), (22, 23, 26)); can.paste(im, (0, pad))
    d = ImageDraw.Draw(can)
    fdir = os.path.join(HERE, "../../fonts")
    fb = ImageFont.truetype(os.path.join(fdir, "WorkSans-Bold.ttf"), 30)
    fs = ImageFont.truetype(os.path.join(fdir, "WorkSans-Bold.ttf"), 19)
    d.text((16, 10), title, font=fb, fill=(235, 235, 235))
    sc = bpy.context.scene
    placed = []
    for n, o in objs.items():
        c = sum((o.matrix_world @ Vector(b) for b in o.bound_box), Vector()) / 8
        p = world_to_camera_view(sc, cam, c)
        x, y = p.x * W, (1 - p.y) * Hh + pad
        lx = x + (60 if x < W * 0.5 else -60)
        ly = y - 46 if n not in ("Lower", "Grip", "Mag", "Round") else y + 40
        while any(abs(ly - q) < 26 and abs(lx - r) < 300 for r, q in placed):
            ly += 28
        placed.append((lx, ly))
        txt = LABELS[n]
        tw = d.textlength(txt, font=fs)
        tx = lx if x < W * 0.5 else lx - tw
        d.line([(x, y), (lx, ly + 11)], fill=(255, 196, 64), width=2)
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 196, 64))
        d.rectangle([tx - 5, ly - 3, tx + tw + 5, ly + 25], fill=(20, 20, 22))
        d.text((tx, ly), txt, font=fs, fill=(240, 240, 240))
    can.save(path.replace(".png", ".jpg"), quality=92)


def renders(objs, mats):
    os.makedirs(REV, exist_ok=True)
    allo = list(objs.values())
    rnd = objs["Round"]
    rnd.hide_render = True                                  # the top round sits inside the magazine lips
    scene.studio(allo)
    bpy.context.scene.cycles.max_bounces = 6
    assign(objs, mats, "FDE")
    scene.frame(allo, yaw=-62, pitch=12, lens=70, fill=0.9, name="hero", aspect=0.56)
    scene.render(f"{REV}/hero_fde.png", 1800, SMP, res_y=1008)
    scene.frame(allo, yaw=-90, pitch=0, lens=135, fill=0.95, name="side", aspect=0.42)
    scene.render(f"{REV}/side_fde.png", 2000, SMP, res_y=840)
    scene.frame(allo, yaw=62, pitch=14, lens=70, fill=0.9, name="left", aspect=0.56)
    scene.render(f"{REV}/left_fde.png", 1800, SMP, res_y=1008)
    for v in ("Black", "Camo"):
        assign(objs, mats, v)
        scene.frame(allo, yaw=-62, pitch=12, lens=70, fill=0.9, name="hero_" + v, aspect=0.56)
        scene.render(f"{REV}/hero_{v.lower()}.png", 1800, SMP, res_y=1008)
    assign(objs, mats, "FDE")
    # close-ups: receiver/port, muzzle + regulator, grip + controls
    for nm, tgt, yaw, pitch, lens in (("close_receiver", ["Lower", "Grip"], -75, 10, 70),
                                      ("close_muzzle", ["Muzzle", "Barrel"], -120, 12, 85),
                                      ("close_grip", ["Grip", "Lower"], -60, 8, 70)):
        sel = [objs[t] for t in tgt]
        if nm == "close_muzzle":
            sel = [objs["Muzzle"]]
        scene.frame(sel, yaw=yaw, pitch=pitch, lens=lens, fill=0.75, name=nm, aspect=0.75)
        scene.render(f"{REV}/{nm}.png", 1400, SMP, res_y=1050)
    exploded(objs, mats)


def exploded(objs, mats):
    """exploded sheet in its own studio (the assembled studio's floor would cut the parts that drop below the gun)"""
    for o in list(bpy.data.objects):
        if o.type in ("LIGHT", "CAMERA") or (o.type == "MESH" and o.name not in {x.name for x in objs.values()} and not o.hide_render):
            bpy.data.objects.remove(o)
    assign(objs, mats, "FDE")
    objs["Round"].hide_render = False
    for n, o in objs.items():
        o.location = EXPLODE[n]
    bpy.context.view_layer.update()
    allo = list(objs.values())
    scene.studio(allo)
    cam = scene.frame(allo, yaw=-90, pitch=6, lens=90, fill=0.86, name="expl", aspect=0.62)
    scene.render(f"{REV}/exploded.png", 2000, SMP, res_y=1240)
    annotate(f"{REV}/exploded.png", cam, objs, "SCAR 17S - exploded (10 game meshes, 18 high-poly parts)")
    for o in objs.values():
        o.location = (0, 0, 0)


def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "work", "scar_low.blend"))
    hc = bpy.data.collections.get("HIGH")
    if hc:
        for o in list(hc.objects):
            bpy.data.objects.remove(o)
    objs = {n: bpy.data.objects["SM_SCAR17S_" + n] for n in PARTS}
    mats = materials()
    if STAGE in ("all", "export"):
        export_all(objs, mats)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "scar17s.blend"), relative_remap=True)
    if STAGE in ("all", "render"):
        renders(objs, mats)
    if STAGE == "exploded":
        exploded(objs, mats)


if __name__ == "__main__":
    main()
