"""DuctTape: builds everything.  .venv/bin/python props/crafting/duct_tape/duct_tape.py [--fast]
Crafting material (splint, barricade patch, grip wrap, armor repair). Roll Ø110 × 48 mm on a 76 mm (3") kraft core,
pivot = roll centre, axis = local X.
Parts: SM_DuctTape (wound tape + spiral-wound core, one closed shell; the end faces carry tape layers outside the
core radius and core plies inside it) + SM_DuctTape_Tail (pulled strip with a torn end; pivot at the top tangent line,
rotate about local X to pull more out)."""
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
RO, RC, RI, WID = 55.0, 41.0, 38.0, 48.0          # tape outer, core outer, core inner, width
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/duct_tape"))
UV_OUT = (0.0, 0.62, 1.0, 0.99)        # outer wrap 345 × 48 mm
UV_END_A = (0.0, 0.0, 0.30, 0.30)
UV_END_B = (0.32, 0.0, 0.62, 0.30)
UV_CORE = (0.0, 0.32, 0.62, 0.60)      # core inside 239 × 48 mm
UV_TAIL = (0.64, 0.0, 1.0, 0.60)


def roll():
    """Lathe around Z (then turned to X): end A (core plies → tape), outer wrap, end B, core inside. One closed shell."""
    h = WID / 2
    end_a = [(RI, -h), (RC, -h), (RO - 0.5, -h)]
    out = [(RO - 0.5, -h), (RO, -h + 0.5), (RO, h - 0.5), (RO - 0.5, h)]
    end_b = [(RO - 0.5, h), (RC, h), (RI, h)]
    core = [(RI, h), (RI, -h)]
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    o = geo.lathe("SM_DuctTape", [dict(pts=sc(end_a), uv=UV_END_A, mode="disc", flip=True), dict(pts=sc(out), uv=UV_OUT),
                                  dict(pts=sc(end_b), uv=UV_END_B, mode="disc"), dict(pts=sc(core), uv=UV_CORE)], segs=48)
    bm = bmesh.new(); bm.from_mesh(o.data)                             # weld the core → end A ring the lathe leaves open
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bm.to_mesh(o.data); bm.free()
    o.data.transform(Matrix.Rotation(math.pi / 2, 4, "Y"))            # roll axis Z → X
    return o


def tail():
    """0.3 mm strip leaving the top of the roll toward +Y, sagging; the end is torn in a zig-zag (real duct tape tears
    along the scrim, so the tear is ragged, not cut)."""
    bm = bmesh.new()
    n, nx = 18, 10
    rng = np.random.default_rng(7)
    jag = [(1.8 if k % 2 else -1.2) + rng.uniform(-0.8, 0.8) for k in range(nx + 1)]
    rows = []
    for i in range(n + 1):
        t = i / n
        y = 85 * t
        z = RO + 0.3 - 34 * t * t
        dzdy = -68 * t / 85
        row = []
        for k in range(nx + 1):
            x = -WID / 2 + WID * k / nx
            yy, zz = y, z
            if i == n:                                                 # torn end: push along the path direction
                d = jag[k] / math.hypot(1, dzdy)
                yy, zz = y + d, z + d * dzdy
            row.append(bm.verts.new((x * MM, yy * MM, zz * MM)))
        rows.append(row)
    for i in range(n):
        for k in range(nx):
            bm.faces.new((rows[i][k], rows[i][k + 1], rows[i + 1][k + 1], rows[i + 1][k]))
    me = bpy.data.meshes.new("SM_DuctTape_Tail"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("SM_DuctTape_Tail", me)
    bpy.context.scene.collection.objects.link(o)
    s = o.modifiers.new("S", "SOLIDIFY"); s.thickness = 0.3 * MM; s.offset = -1.0
    geo.apply_mods(o)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(o.data); bm.free()
    for p in me.polygons:
        p.use_smooth = True
    geo.unwrap_into(o, UV_TAIL)
    geo.set_origin(o, (0, 0, RO * MM))
    return o


def regions(u, v):
    """Per-texel region masks + end-face radius (mm)."""
    out, ea, eb, core, tl = (tex.in_rect(u, v, r) for r in (UV_OUT, UV_END_A, UV_END_B, UV_CORE, UV_TAIL))
    rr = np.zeros_like(u)
    for rect, fl in ((UV_END_A, True), (UV_END_B, False)):
        x, y = tex.disc_map(u, v, rect, RO - 0.5, flip=fl)
        rr = np.where(tex.in_rect(u, v, rect), np.hypot(x, y), rr)
    ends = ea | eb
    tape = out | tl | (ends & (rr > RC))
    board = core | (ends & (rr <= RC))
    return out, ends, core, tl, rr, tape, board


def shared(ao):
    u, v = tex.uv_grid(S)
    out, ends, core, tl, rr, tape, board = regions(u, v)
    # scrim: warp threads along the roll (lines of constant v), sparser weft across; reads through the PE film
    vo = (v - UV_OUT[1]) / (UV_OUT[3] - UV_OUT[1]) * WID
    uo = (u - UV_OUT[0]) / (UV_OUT[2] - UV_OUT[0]) * 2 * math.pi * RO
    warp = np.abs(np.sin(vo / 0.9 * math.pi)) ** 3
    weft = np.abs(np.sin(uo / 2.4 * math.pi)) ** 6
    scrim_o = 0.6 * warp + 0.4 * weft
    tu, tv = (u - UV_TAIL[0]) * S / 12.0, (v - UV_TAIL[1]) * S / 12.0     # tail: ~9 px/mm after the packed unwrap
    scrim_t = 0.6 * np.abs(np.sin(tu / 0.9 * math.pi)) ** 3 + 0.4 * np.abs(np.sin(tv / 2.4 * math.pi)) ** 6
    dents = tex.fbm(S, S, 6, 11) - 0.5
    H = np.where(out, scrim_o * 0.35 + dents * 0.9, 0) + np.where(tl, scrim_t * 0.2, 0)
    # end faces: tape wound in ~0.6 mm visible layers with a few telescoped steps; core in tight kraft plies
    wob = (tex.fbm(S, S, 40, 3) - 0.5) * 0.25
    layers = np.sin((rr + wob) / 0.6 * 2 * math.pi) * 0.5 + 0.5
    steps = np.floor((rr - RC) / 3.3) * 0.12
    plies = np.sin(rr / 0.35 * 2 * math.pi) * 0.5 + 0.5
    lip = np.clip(1 - np.abs(rr - RC) / 0.4, 0, 1)                      # glue line where tape meets the core
    H += np.where(ends & (rr > RC), layers * 0.18 + steps, 0) + np.where(ends & (rr <= RC), plies * 0.12, 0) - np.where(ends, lip * 0.5, 0)
    # core inside: helical paper seam + fibre
    cu = (u - UV_CORE[0]) / (UV_CORE[2] - UV_CORE[0])
    cv = (v - UV_CORE[1]) / (UV_CORE[3] - UV_CORE[1])
    seam = np.clip(1 - np.minimum(((cu + cv * 0.42) % 1.0), 1 - ((cu + cv * 0.42) % 1.0)) / 0.006, 0, 1)
    fibre = tex.fbm(S, S, 120, 5) - 0.5
    H += np.where(core, -seam * 0.6 + fibre * 0.25, 0) + np.where(board & ends, fibre * 0.15, 0)
    # lint and grit stuck to the adhesive edge of the tape ends
    lint = tex.speckle(S, S, 0.0008, 21) * 0.6
    lint = np.where(ends & (rr > RC), np.clip(lint, 0, 1), 0)
    cav = np.where(tape, 0.7 + 0.3 * np.where(ends, layers, 1.0), 1.0) * np.where(core, 1 - seam * 0.5, 1.0) * (1 - lint * 0.4)
    kraft = np.array([0.36, 0.22, 0.11], np.float32)
    base = np.where(board[..., None], kraft * (0.85 + 0.3 * (tex.fbm(S, S, 25, 8)[..., None] - 0.5)), 0.45)
    base = np.where(ends[..., None] & (rr > RC)[..., None], base * (1 - lint[..., None] * 0.6), base)
    rough = np.where(board, 0.88, 0.42).astype(np.float32)
    one = np.ones((S, S), np.float32)
    maskR = tape.astype(np.float32) * (1 - lint * 0.7)
    edge = np.where(ends & (rr > RC), 0.25, 0.0)                        # tape ends read a touch duller than the face
    wear = np.where(out, np.clip(tex.fbm(S, S, 12, 4) * 1.4 - 0.2, 0, 1), 0) * 0.6
    p = lambda k: os.path.join(TEXD, f"T_DuctTape_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.5), S / 345), p("Normal"))
    tex.save(np.stack([ao * cav, np.clip(rough + edge, 0, 1), 0 * one], -1), p("ORM"))
    tex.save(np.stack([maskR, 0 * one, one, wear.astype(np.float32)], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


def print_art(kind, seed):
    """Variant print: woodland camo over every tape surface (camo), and the core's inside print on all variants."""
    u, v = tex.uv_grid(S)
    out, ends, core, tl, rr, tape, board = regions(u, v)
    full = np.zeros((S, S, 4), np.float32)
    if kind == "camo":
        cols = [(0.30, 0.27, 0.15), (0.07, 0.06, 0.05), (0.45, 0.38, 0.22), (0.15, 0.21, 0.10)]
        a, b = tex.fbm(S, S, 9, seed), tex.fbm(S, S, 14, seed + 3)
        idx = np.clip((a * 2.2).astype(int) + (b > 0.62), 0, 3)
        rgb = np.array(cols, np.float32)[idx]
        full = np.concatenate([rgb, tape[..., None].astype(np.float32)], -1)
    from lab.labels import Label
    L = Label(2 * math.pi * RI / 10, WID / 10, S * (UV_CORE[2] - UV_CORE[0]) / (2 * math.pi * RI / 10))
    for k in range(3):
        x = 0.6 + k * 8.0
        L.text(x, 1.4, "ROLL-RITE", "sans", 0.55, "#23324f", "lm")
        L.text(x, 2.5, "ALL-WEATHER 48 mm × 55 m", "sans_r", 0.28, "#23324f", "lm")
        L.text(x, 3.3, "LOT 87-C · KEEP DRY", "mono", 0.22, "#23324f", "lm")
    art = L.array()
    from PIL import Image
    u0, v0, u1, v1 = UV_CORE
    i0, i1, j0, j1 = int((1 - v1) * S), int((1 - v0) * S), int(u0 * S), int(u1 * S)
    patch = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0)), np.float32) / 255
    patch[..., 3] *= 0.85                                                 # print soaks into the kraft
    full[i0:i1, j0:j1] = patch
    return full


VARIANTS = {"Silver": ((0.42, 0.43, 0.44), 0.36, 0.6, None), "Black": ((0.025, 0.025, 0.028), 0.55, 0.0, None),
            "Camo": ((0.20, 0.18, 0.10), 0.6, 0.0, "camo"), "Orange": ((0.85, 0.22, 0.02), 0.45, 0.0, None),
            "White": ((0.80, 0.80, 0.78), 0.45, 0.0, None)}


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
    for i, (k, (colr, rough, metal, kind)) in enumerate(VARIANTS.items()):
        lp = tex.save(print_art(kind, 30 + i), os.path.join(TEXD, f"T_DuctTape_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=colr, RoughA=rough, MetalA=metal, LabelRough=0.6 if kind else 0.85, WearAmount=0.15, Dirt=0.5)
        m = master.material(f"MI_DuctTape_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in (r, t):
        o.data.materials.append(mats["Silver"][0])
    hl = scene.link(export.hull(r, "UCX_SM_DuctTape_00", 48), scene.collection("Collision")); hl.hide_render = True
    tri = {o.name: scene.tris(o) for o in (r, t)}
    export.fbx([r, hl], f"{FBXD}/SM_DuctTape.fbx")
    export.fbx([t], f"{FBXD}/SM_DuctTape_Tail.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "duct_tape.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        flat = i % 2 == 1                                                # alternate: standing on the rim / lying on a face
        for o in ((r,) if flat else (r, t)):
            d = o.copy(); d.data = o.data.copy(); d.data.materials[0] = m
            d.location.x += (i - 2) * 0.13
            if flat:
                d.rotation_euler = (0, math.radians(90), math.radians(25))
                d.location.z += WID / 2 * MM + 0.001
            else:
                d.rotation_euler.z = math.radians(-35)
                d.location.z += RO * MM + 0.001
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
    r.hide_render = t.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=-8, pitch=30, lens=70, fill=0.88, name="Tape", aspect=650 / 1600)
    scene.render(f"{REND}/duct_tape_variants.png", 1600, smp, res_y=650)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "DuctTape", "category": "crafting", "clip": None, "recipes": ["Splint", "BarricadePatch", "GripWrap", "ArmorRepair"],
        "dims_cm": [4.8, 11.0, 11.0], "real_object": "48 mm × 55 m duct tape on a 3\" core", "pivot": "roll centre, axis = local X",
        "parts": {"SM_DuctTape_Tail": {"pivot": "top tangent line (0, 0, 5.5 cm)", "note": "pulled strip with torn end; rotate about X / scale Y to pull"}},
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "collision": "UCX_SM_DuctTape_00"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
