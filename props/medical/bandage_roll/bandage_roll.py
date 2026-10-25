"""BandageRoll: builds everything.  .venv/bin/python props/medical/bandage_roll/bandage_roll.py [--fast]
Contract Bandage_Wrap: roll Ø7 × 5 cm wide, 2.4 cm core, pivot = roll centre, axis = local X.
Parts: SM_BandageRoll (roll: outer wrap, spiral end faces, core hole) + SM_BandageRoll_Tail (loose end flap, pivot at
the tangent line where it leaves the roll; the clip unrolls it about local X)."""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy, bmesh
import numpy as np
from mathutils import Matrix
from lab import scene, master, bake, export, tex, geo

FAST = "--fast" in sys.argv
S = 2048
MM = 0.001
RO, RI, WID = 35.0, 12.0, 50.0
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/bandage_roll"))
UV_OUT = (0.0, 0.55, 1.0, 0.99)        # outer wrap (circumference 220 mm × 50 mm)
UV_END_A = (0.0, 0.05, 0.24, 0.29)
UV_END_B = (0.26, 0.05, 0.50, 0.29)
UV_CORE = (0.0, 0.32, 0.5, 0.52)
UV_TAIL = (0.52, 0.05, 1.0, 0.52)


def roll():
    """Lathe around Z (then turned to X): outer wall, end face A, core, end face B — one closed shell."""
    h = WID / 2
    pts_out = [(RO - 0.8, -h), (RO, -h + 1.0), (RO, h - 1.0), (RO - 0.8, h)]
    end_b = [(RO - 0.8, h), (RI, h)]
    core = [(RI, h), (RI, -h)]
    end_a = [(RI, -h), (RO - 0.8, -h)]
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    o = geo.lathe("SM_BandageRoll", [dict(pts=sc(end_a), uv=UV_END_A, mode="disc"), dict(pts=sc(pts_out), uv=UV_OUT),
                                     dict(pts=sc(end_b), uv=UV_END_B, mode="disc"), dict(pts=sc(core), uv=UV_CORE)], segs=40)
    bm = bmesh.new(); bm.from_mesh(o.data)                             # lathe leaves the core→end_a ring unwelded
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(o.data); bm.free()
    o.data.transform(Matrix.Rotation(math.pi / 2, 4, "Y"))            # roll axis Z → X
    return o


def tail():
    """A 50 mm wide strip leaving the roll tangentially and drooping 60 mm, 0.8 mm thick (closed)."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    n = 16
    path = []
    for i in range(n + 1):
        t = i / n
        a = -math.pi / 2 - 0.15 * t                                    # leaves at the bottom of the roll, curls down
        y, z = 60 * t, -RO - 0.4 - 40 * t * t + 6 * t
        path.append((y, z))
    rows = []
    for i, (y, z) in enumerate(path):
        rows.append([bm.verts.new((x * MM, y * MM, (z + dz) * MM)) for x in (-WID / 2, WID / 2) for dz in (0.0, -0.8)])
    for i in range(n):
        a, b = rows[i], rows[i + 1]
        for (p, q) in ((0, 1), (1, 3), (3, 2), (2, 0)):
            f = bm.faces.new((a[p], a[q], b[q], b[p]))
    bm.faces.new((rows[0][0], rows[0][2], rows[0][3], rows[0][1]))
    bm.faces.new((rows[-1][0], rows[-1][1], rows[-1][3], rows[-1][2]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("SM_BandageRoll_Tail"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("SM_BandageRoll_Tail", me)
    bpy.context.scene.collection.objects.link(o)
    for p in me.polygons:
        p.use_smooth = True
    geo.unwrap_into(o, UV_TAIL)
    geo.set_origin(o, (0, 0, -RO * MM))
    return o


def shared(ao):
    u, v = tex.uv_grid(S)
    out, ea, eb, core, tl = (tex.in_rect(u, v, r) for r in (UV_OUT, UV_END_A, UV_END_B, UV_CORE, UV_TAIL))
    wx = np.abs(np.sin(u * math.pi * 260)) ** 0.5                     # warp + weft threads, open gaps between
    wy = np.abs(np.sin(v * math.pi * 260 * 0.45 * 2.2)) ** 0.5
    weave = np.maximum(wx, wy)
    fray = np.clip(1 - np.minimum(v - UV_OUT[1], UV_OUT[3] - v) / 0.012, 0, 1) * tex.fbm(S, S, 80, 3)
    H = np.where(out | tl, weave * 0.25 + (tex.fbm(S, S, 30, 1) - 0.5) * 0.3 + fray * 0.3, 0)
    cav = np.where(out | tl, 0.55 + 0.45 * weave, 1.0)
    for rect in (UV_END_A, UV_END_B):                                 # spiral of wound layers on the ends
        x, y = tex.disc_map(u, v, rect, RO - 0.8)
        r, a = np.hypot(x, y), np.arctan2(y, x)
        layers = np.sin((r + a / (2 * math.pi) * 0.7) / 0.7 * 2 * math.pi) * 0.5 + 0.5
        H += np.where(tex.in_rect(u, v, rect), layers * 0.2 + (tex.fbm(S, S, 50, 2) - 0.5) * 0.15, 0)
        cav = np.where(tex.in_rect(u, v, rect), 0.6 + 0.4 * layers, cav)
    rough = np.full((S, S), 0.92, np.float32)
    base = np.zeros((S, S, 3), np.float32) + 0.8
    one = np.ones((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_BandageRoll_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.4), S / 220), p("Normal"))
    tex.save(np.stack([ao * cav, rough, 0 * one], -1), p("ORM"))
    tex.save(np.stack([one, 0 * one, one, 0 * one], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


def print_art(kind, seed):
    """Variant print: stripes (elastic), lot text on cohesive, blood stain on the used roll; rest transparent."""
    from lab.labels import Label
    L = Label(22.0, 5.0, S / 22.0)
    if kind == "elastic":
        for k in range(2):
            L.rect(0, 0.4 + k * 3.9, 22, 0.55 + k * 3.9, "#3a5fa0")
    if kind == "clot":
        for k in range(6):
            L.text(1.8 + k * 3.7, 2.5, "CLOTSTOP · HEMOSTATIC", "sans", 0.18, "#7a1010", "mm", rotate=0)
    art = L.array()
    full = np.zeros((S, S, 4), np.float32)
    u0, v0, u1, v1 = UV_OUT
    i0, i1 = int((1 - v1) * S), int((1 - v0) * S)
    from PIL import Image
    full[i0:i1] = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((S, i1 - i0)), np.float32) / 255
    if kind == "used":
        st = np.clip((tex.fbm(S, S, 4, seed) - 0.55) * 4, 0, 1) * np.clip((tex.fbm(S, S, 30, seed + 1) - 0.2) * 3, 0, 1)
        full = np.where(st[..., None] > 0.05, np.concatenate([np.stack([0.42 + 0 * st, 0.05 + 0 * st, 0.04 + 0 * st], -1), st[..., None] * 0.9], -1), full)
    return full


VARIANTS = {"Gauze": ((0.86, 0.85, 0.80), None), "Elastic": ((0.78, 0.62, 0.48), "elastic"), "Cohesive": ((0.20, 0.42, 0.75), None),
            "ClotStop": ((0.55, 0.50, 0.42), "clot"), "Used": ((0.80, 0.77, 0.70), "used")}


def main():
    scene.reset()
    col = scene.collection("Game")
    r, t = scene.link(roll(), col), scene.link(tail(), col)
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (S, 24)
    ao_r = bake.bake(r, [], "AO", n, f"{WORK}/ao.png", samples=smp)
    if n != S:
        from PIL import Image
        ao_r = np.asarray(Image.fromarray((np.clip(ao_r, 0, 1) * 255).astype(np.uint8)).resize((S, S)), np.float32) / 255
    texs = shared(np.where(ao_r < 0.02, 1.0, ao_r))
    mats = {}
    for i, (k, (colr, kind)) in enumerate(VARIANTS.items()):
        lp = tex.save(print_art(kind, i), os.path.join(TEXD, f"T_BandageRoll_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=colr, RoughA=0.92, MetalA=0.0, LabelRough=0.85, WearAmount=0.0, Dirt=0.8 if k == "Used" else 0.6)
        m = master.material(f"MI_BandageRoll_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in (r, t):
        o.data.materials.append(mats["Gauze"][0])
    hl = scene.link(export.hull(r, "UCX_SM_BandageRoll_00", 48), scene.collection("Collision")); hl.hide_render = True
    tri = {o.name: scene.tris(o) for o in (r, t)}
    export.fbx([r, hl], f"{FBXD}/SM_BandageRoll.fbx")
    export.fbx([t], f"{FBXD}/SM_BandageRoll_Tail.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "bandage_roll.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        for o in (r, t):
            d = o.copy(); d.data = o.data.copy(); d.data.materials[0] = m
            d.location.x += (i - 2) * 0.085; d.location.z += RO * MM + 0.001
            d.rotation_euler.z = math.radians(-30)
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
    r.hide_render = t.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=-10, pitch=24, lens=70, fill=0.85, name="Rolls", aspect=650 / 1600)
    scene.render(f"{REND}/bandage_roll_variants.png", 1600, smp, res_y=650)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "BandageRoll", "replaces_rows": ["Bandage", "BandageRoll"], "category": "medical", "clip": "A_Consume_Bandage_Wrap",
        "dims_cm": [5.0, 7.0, 7.0], "real_object": "5 cm conforming / elastic bandage roll", "pivot": "roll centre, axis = local X",
        "parts": {"SM_BandageRoll_Tail": {"pivot": "where the strip leaves the roll (0, 0, -3.5 cm)", "note": "loose end; scale/rotate to unroll"}},
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "collision": "UCX_SM_BandageRoll_00"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
