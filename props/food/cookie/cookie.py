"""Cookie: builds everything.  .venv/bin/python props/food/cookie/cookie.py [--fast]
Top detail (cracks, chips, crumb) comes from cookie_geo.height() evaluated per texel → normal map, so it matches the
silhouette exactly; variants print the bake colour, chips/raisins/oats at the same positions."""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from lab import scene, master, bake, export, tex
import cookie_geo as G

FAST = "--fast" in sys.argv
S = 2048
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/cookie"))


def height_field(x, y):
    r = np.hypot(x, y) / G.R
    h = G.TH * (0.62 + 0.38 * np.clip(1 - r * r, 0, None)) * (1 - 0.25 * r ** 6)
    rn = tex.spectral(*x.shape, 2, 9, 2.2, 21)                     # ridged low-frequency noise → a few big fissures
    rn2 = tex.spectral(*x.shape, 4, 18, 2.0, 22)
    crack = np.maximum(np.clip(1 - np.abs(rn - 0.5) * 22, 0, 1), 0.6 * np.clip(1 - np.abs(rn2 - 0.5) * 30, 0, 1))
    crack *= np.clip((np.hypot(x, y) / G.R - 0.12) * 3, 0, 1)
    h -= 1.2 * crack
    chips = np.zeros_like(x)
    for cx, cy, s in G.CHIPS:
        d = np.hypot(x - cx, y - cy)
        chips = np.maximum(chips, np.clip(1 - (d / s) ** 2, 0, None))
    return h + 1.6 * chips, chips, crack


def shared(ao):
    u, v = tex.uv_grid(S)
    top, bot, side = tex.in_rect(u, v, G.UV_TOP), tex.in_rect(u, v, G.UV_BOT), tex.in_rect(u, v, G.UV_SIDE)
    tx, ty = tex.disc_map(u, v, G.UV_TOP, G.R)
    h, chips, cracks = height_field(tx, ty)
    crumb = tex.fbm(S, S, 60, 3)
    H = np.where(top, h + (crumb - 0.5) * 0.5, 0) + np.where(bot | side, (tex.fbm(S, S, 40, 4) - 0.5) * 0.4, 0)
    rough = np.where(top & (chips > 0.2), 0.35, 0.75).astype(np.float32)
    base = np.zeros((S, S, 3), np.float32) + [0.45, 0.28, 0.12]
    one = np.ones((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_Cookie_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.8), S / (2 * G.R) * 0.7), p("Normal"))
    tex.save(np.stack([ao, rough, 0 * one], -1), p("ORM"))
    tex.save(np.stack([0 * one, 0 * one, one, 0 * one], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}, (chips, cracks)


def art(dough, edge, inclusion, crack_col, fields, inc2=None, sugar=False, seed=0):
    """Variant print: bake gradient (paler centre, browner rim), cracks lighter, inclusions at the chip spots."""
    chips, cracks = fields
    u, v = tex.uv_grid(S)
    tx, ty = tex.disc_map(u, v, G.UV_TOP, G.R)
    r = np.clip(np.hypot(tx, ty) / G.R, 0, 1)
    c = lambda h: np.array(tex.srgb_to_lin(tex.hex2rgb(h)))
    n = tex.fbm(S, S, 20, seed)
    col = c(dough) * (1 - r[..., None] ** 3) + c(edge) * r[..., None] ** 3
    col = col * (0.85 + 0.3 * n[..., None])
    col = col * (1 - 0.5 * cracks[..., None]) + c(crack_col) * 0.5 * cracks[..., None]
    k = np.clip(chips * 2, 0, 1)
    col = col * (1 - k[..., None]) + c(inclusion) * k[..., None]
    if inc2:                                                          # rolled-oat flakes scattered over the top
        oats = np.clip(tex.speckle(S, S, 0.0025, seed + 7, 4.0) * 1.5, 0, 1)
        col = col * (1 - oats[..., None]) + c(inc2) * oats[..., None]
    if sugar:
        sp = tex.speckle(S, S, 0.01, seed + 1, 1.0)
        col = col * (1 - sp[..., None]) + 0.9 * sp[..., None]
    side_bot = ~tex.in_rect(u, v, G.UV_TOP)
    col = np.where(side_bot[..., None], c(edge) * (0.8 + 0.3 * n[..., None]), col)
    return np.concatenate([tex.lin_to_srgb(col), np.ones((S, S, 1), np.float32)], -1)


VARIANTS = {"ChocChip": dict(dough="#d9a35a", edge="#9a5a22", inclusion="#2a1408", crack_col="#f0cc8a"),
            "OatmealRaisin": dict(dough="#c99a5e", edge="#8a5a2a", inclusion="#3a1a14", inc2="#e8d2a0", crack_col="#e8c890"),
            "DoubleChoc": dict(dough="#4a2414", edge="#2a120a", inclusion="#1a0a04", crack_col="#6a3a20"),
            "SugarCookie": dict(dough="#efd29a", edge="#c9944a", inclusion="#efd29a", crack_col="#f8e6c0", sugar=True)}


def main():
    scene.reset()
    o = scene.link(G.cookie("SM_Cookie"), scene.collection("Game"))
    bm = __import__("bmesh").new(); bm.from_mesh(o.data)
    __import__("bmesh").ops.triangulate(bm, faces=bm.faces); bm.to_mesh(o.data); bm.free()
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (S, 32)
    ao = bake.bake(o, [], "AO", n, f"{WORK}/ao.png", samples=smp)
    if n != S:
        from PIL import Image
        ao = np.asarray(Image.fromarray((np.clip(ao, 0, 1) * 255).astype(np.uint8)).resize((S, S)), np.float32) / 255
    texs, fields = shared(np.where(ao < 0.02, 1.0, ao))
    mats = {}
    for i, (k, a) in enumerate(VARIANTS.items()):
        lp = tex.save(art(fields=fields, seed=i, **a), os.path.join(TEXD, f"T_Cookie_Label_{k}.png"), "RGBA")
        prm = dict(LabelRough=0.72, WearAmount=0.0, Dirt=0.15)
        m = master.material(f"MI_Cookie_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    o.data.materials.clear(); o.data.materials.append(mats["ChocChip"][0])
    h = scene.link(export.hull(o, "UCX_SM_Cookie_00", 48), scene.collection("Collision")); h.hide_render = True
    tri = scene.tris(o)
    export.fbx([o, h], f"{FBXD}/SM_Cookie.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "cookie.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        d = o.copy(); d.data = o.data.copy(); d.data.materials[0] = m
        d.rotation_euler = (math.pi / 2, 0, i * 0.9)                    # lay flat, top up
        d.location = ((i - 1.5) * 0.085, 0, 0.0075)
        bpy.context.scene.collection.objects.link(d)
        dups.append(d)
    o.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=0, pitch=48, lens=70, fill=0.85, name="Cookies", aspect=700 / 1600)
    scene.render(f"{REND}/cookie_variants.png", 1600, smp, res_y=700)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "Cookie", "replaces_rows": ["Cookie"], "category": "food", "clip": "A_Consume_Eat_Disc",
        "dims_cm": [7.26, 1.63, 7.23], "real_object": "bakery cookie, Ø73 × 15 mm", "pivot": "centre, disc axis = Y (top faces +Y in Blender / -Y in UE)",
        "tris": {"SM_Cookie": tri}, "textures": {"size": S, "normal": "OpenGL (flip green in UE)",
                                                 "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "collision": "UCX_SM_Cookie_00"})
    print("TRIS", {"SM_Cookie": tri})


if __name__ == "__main__":
    main()
