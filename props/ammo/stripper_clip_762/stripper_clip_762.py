"""7.62x51 five-round stripper clip (stamped steel channel, rim lips, end stops, flat follower spring). Builds everything.
    .venv/bin/python props/ammo/stripper_clip_762/stripper_clip_762.py [--fast]
SM_StripperClip762: pivot bottom centre, channel along +Y, rounds stand on +Z with their rims in the channel.
The rounds are not part of the mesh: manifest slots_cm place SM_Round762_Low (the hero render does that).
Skins: Phosphate (parkerized grey-green), Blued, Brass-plated."""
import os, sys, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, os.path.dirname(HERE))
import bpy, bmesh
import numpy as np
from mathutils import Matrix
from lab import scene, master, bake, export, tex, geo
import box_lib

FAST = "--fast" in sys.argv
MM = 0.001
S = 512
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/stripper_clip"))
N, PITCH = 5, 12.1
L = N * PITCH + 1.0
RIM, RH, RG = 5.97, 1.37, 4.90           # 7.62x51 rim radius, rim height, groove radius (round_762)


def channel():
    t = 0.5
    wi = RIM + 0.15
    # cross-section (x, z) mm, ordered closed polygon: bottom right → up right wall → lip → down inside → floor → left side mirrored
    poly = [(wi + t, -t), (wi + t, RH + 0.75), (RG + 0.4, RH + 0.75), (RG + 0.4, RH + 0.75 - t), (wi, RH + 0.75 - t), (wi, 0.0),
            (-wi, 0.0), (-wi, RH + 0.75 - t), (-RG - 0.4, RH + 0.75 - t), (-RG - 0.4, RH + 0.75), (-wi - t, RH + 0.75), (-wi - t, -t)]
    bm = bmesh.new()
    front = [bm.verts.new((x * MM, -L / 2 * MM, z * MM)) for x, z in poly]
    back = [bm.verts.new((x * MM, L / 2 * MM, z * MM)) for x, z in poly]
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((front[i], front[j], back[j], back[i]))
    bm.faces.new(front[::-1]); bm.faces.new(back)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4], quad_method="BEAUTY")
    me = bpy.data.meshes.new("SM_StripperClip762")
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(me.name, me)
    bpy.context.scene.collection.objects.link(o)
    # end stops (tabs bent up at both ends) and the flat spring strip on the floor
    for y in (-L / 2 + 0.4, L / 2 - 0.4):
        tab = geo.rounded_box("tab", 7.0, 0.8, RH + 1.9 + 0.5, 0.0, -0.5, 0.0, center=(0, y))
        geo.boolean(o, tab, "UNION")
    sp = geo.rounded_box("spring", 6.0, L - 6.0, 0.35, 0.0, -0.1, 0.0, center=(0, 0))
    geo.boolean(o, sp, "UNION")
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4], quad_method="BEAUTY")
    bm.to_mesh(o.data); bm.free()
    for p in o.data.polygons:
        p.use_smooth = False
    return o


VARIANTS = {"Phosphate": dict(ColorA=(0.09, 0.095, 0.085), RoughA=0.7, MetalA=0.6),
            "Blued": dict(ColorA=(0.04, 0.045, 0.06), RoughA=0.35, MetalA=1.0),
            "Brass": dict(ColorA=(0.70, 0.50, 0.20), RoughA=0.32, MetalA=1.0)}


def main():
    scene.reset()
    col = scene.collection("Game")
    o = scene.link(channel(), col)
    geo.unwrap_into(o, (0.01, 0.01, 0.99, 0.99), margin=0.02)
    geo.weighted_normals(o)
    os.makedirs(WORK, exist_ok=True)
    ao = bake.bake(o, [], "AO", S, f"{WORK}/ao.png", samples=8 if FAST else 32)
    ao = np.where(ao < 0.02, 1.0, ao)
    hi = o.copy(); hi.data = o.data.copy(); bpy.context.scene.collection.objects.link(hi)
    for m in list(hi.modifiers):
        hi.modifiers.remove(m)
    sd = hi.modifiers.new("Sub", "SUBSURF"); sd.subdivision_type = "SIMPLE"; sd.levels = sd.render_levels = 3
    cv = bake.bake(o, [hi], "CURV", S, f"{WORK}/cv.png", samples=1, extrusion=0.0003, max_ray=0.001)
    bpy.data.objects.remove(hi)
    cv = np.where(cv < 0.01, 0.5, cv)
    n1, n2 = tex.fbm(S, S, 10, 1), tex.fbm(S, S, 40, 2)
    base = np.zeros((S, S, 3), np.float32) + np.array((0.42, 0.42, 0.42), np.float32) * (0.85 + 0.3 * n1[..., None])
    one = np.ones((S, S), np.float32)
    wear = np.clip(tex.remap(cv, 0.55, 0.7) * (0.5 + n2) , 0, 1)
    p = lambda k: os.path.join(TEXD, f"T_StripperClip762_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal((n2 - 0.5) * 0.02 + tex.scratches(S, S, 120, (0.01, 0.05), 1, 3) * -0.04, 4.0), p("Normal"))
    tex.save(np.stack([ao, 0.3 + 0.1 * n1, one], -1), p("ORM"))
    tex.save(np.stack([one, 0 * one, 0 * one, wear], -1), p("Mask"), "RGBA")
    texs = {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
    mats = {}
    for k, v in VARIANTS.items():
        prm = dict(v, LabelRough=0.5, WearAmount=0.4, Dirt=0.5)
        m = master.material(f"MI_StripperClip762_{k}", texs, prm)
        m.use_fake_user = True
        mats[k] = (m, prm)
    if o.data.materials:
        o.data.materials[0] = mats["Phosphate"][0]
    else:
        o.data.materials.append(mats["Phosphate"][0])
    h = scene.link(export.hull(o, "UCX_SM_StripperClip762_00", 24), scene.collection("Collision")); h.hide_render = True
    tri = {o.name: scene.tris(o)}
    export.fbx([o, h], f"{FBXD}/SM_StripperClip762.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "stripper_clip_762.blend"), relative_remap=True)
    slots = [(0.0, (i - (N - 1) / 2) * PITCH, 0.0) for i in range(N)]
    # hero: three skins; the first loaded with five rounds, the others empty, one lying on its side
    o.hide_render = True; h.hide_render = True
    rnd = box_lib.load_round(dict(round=("round_762", "Round762_Low", "FMJ")))
    objs = []
    for i, (k, (m, _)) in enumerate(mats.items()):
        d = o.copy(); d.data = o.data.copy(); d.data.materials[0] = m; d.hide_render = False
        bpy.context.scene.collection.objects.link(d)
        M = Matrix.Translation(((i - 0.6) * 0.03 + (0.0 if i == 0 else 0.012), 0.004 * i, 0.0005)) @ Matrix.Rotation(math.radians(8 * i), 4, "Z")
        d.matrix_world = M
        objs.append(d)
        if i == 0:
            for x, y, z in slots:
                r = rnd.copy(); r.hide_render = False
                bpy.context.scene.collection.objects.link(r)
                r.matrix_world = M @ Matrix.Translation((x * MM, y * MM, z * MM)) @ Matrix.Rotation(i * 0.3 + y, 4, "Z")
                objs.append(r)
    scene.studio(objs)
    scene.frame(objs, yaw=-25, pitch=24, lens=70, fill=0.9, name="Hero", aspect=576 / 1024)
    scene.render(os.path.join(REND, "stripper_clip_762.png"), 1024, 16 if FAST else 48, res_y=576)
    lo, hi_ = scene.bounds([o])
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "StripperClip762", "replaces_rows": ["StripperClip (new)"], "category": "ammo", "clip": None,
        "real_object": "5-round 7.62x51 / M14-pattern stamped steel stripper clip, 6.2 x 1.4 x 0.3 cm",
        "dims_cm": [round((hi_[i] - lo[i]) * 100, 2) for i in range(3)], "pivot": "bottom centre, channel along +Y",
        "contents": {"round_mesh": "SM_Round762_Low", "count": N, "slots_cm": [[x / 10, y / 10, z / 10] for x, y, z in slots]},
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)"},
        "variant_params": {k: v[1] for k, v in mats.items()}, "mask": "R finish · A edge wear",
        "collision": "UCX_SM_StripperClip762_00", "lods": "none"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
