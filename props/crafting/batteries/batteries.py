"""Batteries: builds everything.  .venv/bin/python props/crafting/batteries/batteries.py [--fast]
Crafting material (flashlight, radio, NVG, keycard-reader bypass). AA ×4 blister card + a loose AA cell.
Parts: SM_Batteries (printed card with euro hang slot + four AA cells; pivot = card bottom centre, print faces -Y),
       SM_Batteries_Blister (PET bubble, cup open at the back, glass master), SM_Battery_AA (one cell, pivot = - end,
       + nub along +Z). Cells share one UV island (identical prints by design): QC --uv-overlap-ok SM_Batteries."""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from mathutils import Matrix
from PIL import Image
from lab import scene, master, bake, export, tex, geo
from lab.labels import Label

FAST = "--fast" in sys.argv
S = 2048
MM = 0.001
CW, CH, CT = 85.0, 140.0, 0.8                    # card
BR, BL = 7.25, 50.5                              # AA cell radius / length
BX = (-22.2, -7.4, 7.4, 22.2)
BZ = 36.0
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/batteries"))
UV_WRAP = (0.0, 0.0, 0.30, 0.33)
UV_BOT, UV_TOP = (0.32, 0.0, 0.42, 0.10), (0.32, 0.12, 0.42, 0.22)
FRONT, BACK = (0.0, 0.36, 0.40, 0.99), (0.42, 0.36, 0.82, 0.99)
DIELINE = {"-Y": (FRONT, (-CW / 2, CW / 2), (0, CH)), "+Y": (BACK, (-CW / 2, CW / 2), (0, CH)),
           "-X": ((0.84, 0.36, 0.85, 0.99), (-CT / 2, CT / 2), (0, CH)), "+X": ((0.86, 0.36, 0.87, 0.99), (-CT / 2, CT / 2), (0, CH)),
           "+Z": ((0.0, 0.345, 0.40, 0.352), (-CW / 2, CW / 2), (-CT / 2, CT / 2)), "-Z": ((0.42, 0.345, 0.82, 0.352), (-CW / 2, CW / 2), (-CT / 2, CT / 2))}
UV_INNER = (0.88, 0.36, 0.99, 0.47)
UV_BLISTER = (0.50, 0.0, 0.80, 0.30)


def cell(name):
    """AA cell: flat - end with insulator ring, wrapped can, crimped shoulder, + nub."""
    bot = [(0, 0), (4.6, 0), (5.4, 0.15), (6.6, 0.25), (7.0, 0.5)]
    side = [(7.0, 0.5), (BR, 1.0), (BR, BL - 1.4), (7.0, BL - 0.9)]
    top = [(7.0, BL - 0.9), (5.6, BL - 0.8), (2.9, BL - 0.75), (2.7, BL - 0.6), (2.6, BL - 0.1), (2.3, BL), (0, BL)]
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    return geo.lathe(name, [dict(pts=sc(bot), uv=UV_BOT, mode="disc", flip=True), dict(pts=sc(side), uv=UV_WRAP),
                            dict(pts=sc(top), uv=UV_TOP, mode="disc")], segs=24)


def card():
    c = geo.rounded_box("SM_Batteries", CW, CH, CT, 3.0, -CT / 2, 0.0, segs=4)       # built flat, stood up below
    slot = geo.rounded_box("slot", 15, 5.5, 4, 2.7, -2, 0.0, segs=4, center=(0, CH / 2 - 9))
    geo.boolean(c, slot)
    c.data.transform(Matrix.Translation((0, CH / 2 * MM, 0)))
    c.data.transform(Matrix.Rotation(math.pi / 2, 4, "X"))                           # XY card → XZ, print side -Y
    geo.box_uv(c, DIELINE, (0, 0, CH / 2 * MM), UV_INNER)
    return c


def blister():
    """PET bubble: the outer shape with its back face removed, solidified 0.4 mm inward (one closed shell, rim on the card)."""
    import bmesh
    b = geo.rounded_box("SM_Batteries_Blister", 66, 17.6, 60, 5.0, BZ - 4, 3.0, segs=4, center=(0, -CT / 2 - 8.8 + 0.05))
    bm = bmesh.new(); bm.from_mesh(b.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.normal.y > 0.99 and f.calc_center_median().y > -CT / 2 * MM - 0.002], context="FACES")
    bm.to_mesh(b.data); bm.free()
    sm = b.modifiers.new("S", "SOLIDIFY"); sm.thickness = 0.4 * MM; sm.offset = -1.0
    geo.apply_mods(b)
    geo.unwrap_into(b, UV_BLISTER)
    return b


def pack():
    cd = card()
    cells = []
    for x in BX:
        k = cell("c")
        k.location = (x * MM, (-CT / 2 - BR + 0.5) * MM, BZ * MM)           # seated 0.5 mm into the card (survives LOD2)
        geo.bake_xform(k)
        cells.append(k)
    return geo.join([cd] + cells, "SM_Batteries")


def paste(full, art, rect):
    u0, v0, u1, v1 = rect
    i0, i1, j0, j1 = int((1 - v1) * S), int((1 - v0) * S), int(u0 * S), int(u1 * S)
    im = Image.fromarray((np.clip(art, 0, 1) * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0), Image.LANCZOS)
    full[i0:i1, j0:j1] = np.asarray(im, np.float32) / 255


def shared(ao):
    u, v = tex.uv_grid(S)
    wrap, bot, top = (tex.in_rect(u, v, r) for r in (UV_WRAP, UV_BOT, UV_TOP))
    cardr = tex.in_rect(u, v, (0.0, 0.34, 0.99, 0.99))
    rr = np.zeros_like(u)
    for rect, fl in ((UV_BOT, True), (UV_TOP, False)):
        x, y = tex.disc_map(u, v, rect, BR, flip=fl)
        rr = np.where(tex.in_rect(u, v, rect), np.hypot(x, y), rr)
    steel = (bot & (rr < 4.6)) | (top & (rr < 2.95))
    insul = (bot & (rr >= 4.6) & (rr < 5.6)) | (top & (rr >= 2.95) & (rr < 5.6))
    fold = (bot | top) & (rr >= 5.6)                                     # wrap folded over the shoulders
    # normals: brushed rings + stamped ring on the - end, nub dome, wrap seam, paperboard fibre
    H = np.where(steel, np.sin(rr / 0.08) * 0.03 + np.where(bot & (np.abs(rr - 3.2) < 0.15), -0.4, 0), 0)
    H += np.where(fold, (tex.fbm(S, S, 60, 2) - 0.5) * 0.3 + np.sin(np.arctan2(*tex.disc_map(u, v, UV_TOP, BR)) * 24) * 0.12, 0)
    wu = (u - UV_WRAP[0]) / (UV_WRAP[2] - UV_WRAP[0])
    H += np.where(wrap, np.clip(1 - np.abs(wu - 0.985) / 0.006, 0, 1) * 0.5 + (tex.fbm(S, S, 90, 4) - 0.5) * 0.04, 0)
    H += np.where(cardr, (tex.fbm(S, S, 200, 6) - 0.5) * 0.012, 0)
    base = np.full((S, S, 3), 0.80, np.float32)
    base = np.where(steel[..., None], np.float32(0.62), base)
    base = np.where(insul[..., None], np.float32(0.03), base)
    rough = np.where(steel, 0.28, np.where(insul, 0.6, 0.55)).astype(np.float32)
    metal = steel.astype(np.float32)
    maskR = (wrap | fold).astype(np.float32)                             # metallised wrap film (per brand)
    wear = np.where(wrap | fold, np.clip(tex.fbm(S, S, 20, 9) - 0.35, 0, 1) * 1.5, np.where(cardr, tex.fbm(S, S, 10, 3) * 0.7, 0))
    p = lambda k: os.path.join(TEXD, f"T_Batteries_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.4), S / 46), p("Normal"))
    tex.save(np.stack([ao, rough, metal], -1), p("ORM"))
    one = np.ones((S, S), np.float32)
    tex.save(np.stack([maskR, 0 * one, (wrap | fold | cardr).astype(np.float32), np.clip(wear, 0, 1)], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


BRANDS = {
    "KiloWatt": dict(name="KILOWATT", c1="#1b2b66", c2=None, ink="#f2c230", tag="POWER THAT KEEPS ON", film=(0.62, 0.66, 0.72), font="display"),
    "BodegaBasics": dict(name="BODEGA BASICS", c1="#c4161c", c2="#f4f1ea", ink="#c4161c", tag="GOOD ENOUGH. CHEAP.", film=(0.80, 0.80, 0.78), font="heavy"),
    "Ironcell": dict(name="IRONCELL", c1="#111111", c2=None, ink="#c9a227", tag="FIELD GRADE · -40° TO 60°", film=(0.75, 0.55, 0.20), font="stencil"),
}


def wrap_art(b):
    """4.56 × 5.05 cm wrap. Opaque ink on the brand band; film (bare metallic ColorA) shows where alpha = 0."""
    L = Label(2 * math.pi * BR / 10, (BL - 2.4) / 10, 420)
    w, h = L.w, L.h
    L.rect(0, 0, w, h * 0.42, b["c1"])
    if b["c2"]:
        L.rect(0, h * 0.42, w, h, b["c2"])
    L.text(w * 0.25, h * 0.71, b["name"], b["font"], 0.42, b["ink"] if b["c2"] else "#f4f1ea", "mm", rotate=90, max_w=h * 0.5)
    L.text(w * 0.75, h * 0.21, b["name"], b["font"], 0.36, b["ink"], "mm", rotate=90, max_w=h * 0.38)
    L.text(w * 0.5, h * 0.06, "+", "sans", 0.4, "#f4f1ea", "mm")
    L.text(w * 0.52, h * 0.70, "AA · 1.5V · LR6 · ALKALINE", "mono_b", 0.13, "#222222", "mm", rotate=90)
    L.text(w * 0.62, h * 0.93, "EXP 03-2031", "mono", 0.1, "#333333", "mm")
    return L.array()


def card_front(b):
    L = Label(CW / 10, CH / 10, 220)
    w, h = L.w, L.h
    L.gradient(0, 0, w, h, b["c1"], "#06080f" if not b["c2"] else b["c2"])
    L.rect(0, 0, w, 1.6, "#000000" if not b["c2"] else b["c1"])
    L.text(w / 2, 2.6, b["name"], b["font"], 0.95, b["ink"] if not b["c2"] else "#ffffff", "mm", max_w=w - 0.8, stroke=0.03, stroke_color="#000000")
    L.text(w / 2, 3.45, b["tag"], "sans", 0.28, "#f4f1ea" if not b["c2"] else "#c4161c", "mm", tracking=0.05)
    L.rect(0.6, 4.4, w - 0.6, 10.6, "#2a3c80" if b["name"] == "KILOWATT" else "#e9e4d8" if b["c2"] else "#1e1e1e", radius=0.5)
    L.burst(7.0, 4.7, 0.7, 0.95, 16, b["ink"])
    L.text(7.0, 4.6, "10 YR", "heavy", 0.24, b["c1"], "mm")
    L.text(7.0, 4.9, "STORAGE", "sans", 0.12, b["c1"], "mm")
    L.rect(0, h - 2.6, w, h, b["ink"])
    L.text(0.5, h - 1.3, "AA", "heavy", 0.9, b["c1"] if not b["c2"] else "#ffffff", "lm")
    L.text(2.4, h - 1.6, "×4  ALKALINE", "sans", 0.35, b["c1"] if not b["c2"] else "#ffffff", "lm")
    L.text(2.4, h - 0.95, "1.5V  ·  LR6  ·  MN1500 SIZE", "mono_b", 0.2, b["c1"] if not b["c2"] else "#ffffff", "lm")
    return L.array()


def card_back(b, seed):
    L = Label(CW / 10, CH / 10, 220)
    w, h = L.w, L.h
    L.rect(0, 0, w, h, "#ece8de")
    L.rect(0, 0, w, 1.1, b["c1"])
    L.text(w / 2, 1.9, b["name"], b["font"], 0.5, b["c1"], "mm")
    L.para(0.6, 2.8, w - 1.2, "CAUTION: Do not recharge, heat, crush or dispose of in fire. Insert correctly (+ and −). "
           "Replace all cells at the same time. Keep away from small children; if swallowed, get medical help at once. "
           "Remove from devices you will not use for a while. Store cool and dry.", "sans_r", 0.2, "#222222")
    L.para(0.6, 7.6, w - 1.2, "Packed in the city by Five Boroughs Supply Co. Not affiliated with any transit authority. "
           "Recycle at participating bodegas.", "sans_r", 0.17, "#444444")
    L.barcode(1.2, h - 3.2, 4.2, 1.8, f"07{seed:02d}93118425"[:13])
    L.text(w - 0.7, h - 0.7, "LOT 2X-118", "mono", 0.16, "#333333", "rm")
    from lab.labels import age
    return age(L.array(), seed)


def print_art(b, seed):
    full = np.zeros((S, S, 4), np.float32)
    wa = wrap_art(b)
    if not b["c2"]:                                                      # no second band: lower half is bare film
        hh = wa.shape[0]
        wa[int(hh * 0.42):, :, 3] *= np.where(wa[int(hh * 0.42):, :, :3].sum(-1) > 0.05, 1.0, 0.0)
    paste(full, wa, (UV_WRAP[0], UV_WRAP[1] + 0.33 * 1.2 / BL, UV_WRAP[2], UV_WRAP[3] - 0.33 * 1.2 / BL))
    paste(full, card_front(b), FRONT)
    paste(full, card_back(b, seed), BACK)
    return full


def main():
    scene.reset()
    col = scene.collection("Game")
    pk, bl = scene.link(pack(), col), scene.link(blister(), col)
    one = scene.link(cell("SM_Battery_AA"), col)
    one.location.x = 0.12
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (S, 24)
    ao = bake.bake(pk, [bl], "AO", n, f"{WORK}/ao.png", samples=smp)
    if n != S:
        ao = np.asarray(Image.fromarray((np.clip(ao, 0, 1) * 255).astype(np.uint8)).resize((S, S)), np.float32) / 255
    texs = shared(np.where(ao < 0.02, 1.0, ao))
    gm = master.glass("M_Batteries_Blister", tint=(0.96, 0.98, 0.97), rough=0.06)
    mats = {}
    for i, (k, b) in enumerate(BRANDS.items()):
        lp = tex.save(print_art(b, i), os.path.join(TEXD, f"T_Batteries_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=b["film"], RoughA=0.22, MetalA=0.9, LabelRough=0.4, WearAmount=0.2, Dirt=0.4)
        m = master.material(f"MI_Batteries_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in (pk, one):
        o.data.materials.append(mats["KiloWatt"][0])
    bl.data.materials.append(gm)
    cols = scene.collection("Collision")
    hulls = [scene.link(export.hull(pk, "UCX_SM_Batteries_00", 32), cols), scene.link(export.hull(one, "UCX_SM_Battery_AA_00", 16), cols)]
    for hh in hulls:
        hh.hide_render = True
    one.location.x = 0
    l1, l2 = export.lod(pk, 0.5, "SM_Batteries_LOD1"), export.lod(pk, 0.25, "SM_Batteries_LOD2")
    for o in (l1, l2):
        o.hide_render = True; o.hide_set(True)
    tri = {o.name: scene.tris(o) for o in (pk, bl, one, l1, l2)}
    export.fbx([pk, hulls[0]], f"{FBXD}/SM_Batteries.fbx")
    export.fbx([bl], f"{FBXD}/SM_Batteries_Blister.fbx")
    export.fbx([l1], f"{FBXD}/SM_Batteries_LOD1.fbx")
    export.fbx([l2], f"{FBXD}/SM_Batteries_LOD2.fbx")
    hulls[1].location.x = 0
    export.fbx([one, hulls[1]], f"{FBXD}/SM_Battery_AA.fbx")
    one.location.x = 0.12
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "batteries.blend"), relative_remap=True)
    s = bpy.context.scene
    s.cycles.transmission_bounces = s.cycles.max_bounces = 8
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        for o in (pk, bl):
            d = o.copy(); d.data = o.data.copy()
            if o is pk:
                d.data.materials[0] = m
            d.location = ((i - 1.3) * 0.105, 0, 0.0005)
            d.rotation_euler.z = math.radians(-12 + 12 * i)
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
        for j in range(2):                                                # loose cells in front
            d = one.copy(); d.data = one.data.copy(); d.data.materials[0] = m
            d.rotation_euler = (0, math.radians(90), math.radians(10 + 40 * j + 30 * i))
            d.location = ((i - 1.3) * 0.105 + (j - 0.5) * 0.035, -0.07 - 0.015 * j, BR * MM)
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
    for o in (pk, bl, one):
        o.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=-6, pitch=14, lens=70, fill=0.9, name="Batt", aspect=900 / 1600)
    scene.render(f"{REND}/batteries_variants.png", 1600, 16 if FAST else 128, res_y=900)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "Batteries", "category": "crafting", "clip": None, "recipes": ["Flashlight", "Radio", "NVG", "KeycardBypass"],
        "dims_cm": {"SM_Batteries": [8.5, 1.9, 14.0], "SM_Battery_AA": [1.45, 1.45, 5.05]},
        "pivot": "card bottom centre, print faces -Y; cell: - end, + toward +Z",
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "glass": {"SM_Batteries_Blister": "M_Batteries_Blister"},
        "collision": ["UCX_SM_Batteries_00", "UCX_SM_Battery_AA_00"],
        "lods": ["SM_Batteries_LOD1 (50%)", "SM_Batteries_LOD2 (25%)"]})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
