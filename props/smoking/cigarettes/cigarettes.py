"""Cigarettes (hard pack + flip lid, single cigarette) and a disposable lighter: builds everything.
    .venv/bin/python props/smoking/cigarettes/cigarettes.py [--fast]
One texture set (2048): the pack dieline, cigarette paper, lighter. Pack variants = brands; lighter variants = colours."""
import os, sys, math, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from PIL import Image
from lab import scene, master, bake, export, tex
from lab.labels import Label, age
import cig_geo as G

FAST = "--fast" in sys.argv
S = 2048
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/cigarettes"))
WARN = "SURGEON GENERAL'S WARNING: Smoking Causes Lung Cancer, Heart Disease, Emphysema, And May Complicate Pregnancy."


def paste(full, art, rect, rot=0):
    if rot:
        art = np.ascontiguousarray(np.rot90(art, rot))
    u0, v0, u1, v1 = rect
    j0, j1, i0, i1 = int(round(u0 * S)), int(round(u1 * S)), int(round((1 - v1) * S)), int(round((1 - v0) * S))
    full[i0:i1, j0:j1] = np.asarray(Image.fromarray((np.clip(art, 0, 1) * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0), Image.LANCZOS), np.float32) / 255


def shared(ao):
    u, v = tex.uv_grid(S)
    pack = np.zeros((S, S), bool)
    for r in (G.FRONT, G.BACK, G.LEFT, G.RIGHT, G.TOP, G.BOTTOM):
        pack |= tex.in_rect(u, v, r)
    inner, cig = tex.in_rect(u, v, (0.70, 0.25, 0.99, 0.386)), tex.in_rect(u, v, G.UV_CIG)
    metal = tex.in_rect(u, v, (0.32, 0.0, 0.99, 0.22)) & ~tex.in_rect(u, v, (0.32, 0.0, 0.64, 0.22)) | tex.in_rect(u, v, (0.81, 0.0, 0.99, 0.10))
    base = np.where(pack[..., None], 0.85, np.where(inner[..., None], [0.8, 0.8, 0.82], 0.9))
    met = np.where(inner | metal, 1.0, 0.0).astype(np.float32)
    rough = np.where(pack, 0.35, np.where(inner, 0.3, np.where(metal, 0.3, 0.6))).astype(np.float32)
    H = np.where(pack, (tex.spectral(S, S, 100, 600, 1.0, 1) - 0.5) * 0.02, 0) + \
        np.where(inner, (tex.spectral(S, S, 10, 120, 1.8, 2) - 0.5) * 0.15, 0) + \
        np.where(tex.in_rect(u, v, (0.81, 0.0, 0.99, 0.10)), (np.abs(np.sin(u * 2 * math.pi * 260)) ** 0.5) * 0.08, 0)   # wheel knurl
    wear = np.where(pack, np.clip(tex.scratches(S, S, 80, (0.004, 0.02), 1, 3) * 0.8 + (tex.fbm(S, S, 40, 4) > 0.8) * 0.5, 0, 1), 0)
    one = np.ones((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_Cigarettes_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.4), S / 160), p("Normal"))
    tex.save(np.stack([ao, rough, met], -1), p("ORM"))
    mR = tex.in_rect(u, v, (0.32, 0.0, 0.64, 0.22)).astype(np.float32)
    tex.save(np.stack([mR, 0 * one, (pack | cig).astype(np.float32), wear], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


def brand_print(name, sub, bg, ink, accent, logo, filt="#c98a3a", seed=0):
    full = np.zeros((S, S, 4), np.float32)
    F = Label(5.5, 8.8, 140, bg=bg)                             # front: lid band (top 1.8 cm) + brand panel
    F.rect(0, 0, 5.5, 1.8, accent)
    F.text(2.75, 0.9, name.split()[0], "sans", 0.42, bg, "mm", tracking=0.25)
    logo(F)
    F.text(2.75, 5.6, name, "display", 0.85, ink, "mm", tracking=0.04, max_w=5.0)
    F.text(2.75, 6.4, sub, "sans", 0.26, accent, "mm", tracking=0.2)
    F.text(2.75, 8.25, "20 CLASS A CIGARETTES", "sans", 0.2, ink, "mm", tracking=0.12)
    B = Label(5.5, 8.8, 140, bg=bg)                             # back: the warning box, barcode
    B.rect(0, 0, 5.5, 1.8, accent)
    B.rect(0.35, 2.3, 5.15, 5.9, "#ffffff", outline="#111111", width=0.05)
    B.para(0.6, 2.6, 4.3, WARN, "sans", 0.3, "#111111", 1.2)
    B.barcode(1.6, 6.5, 2.3, 1.0, "0" + str(zlib.crc32(name.encode()) % 10 ** 11).zfill(11))
    Ls = Label(2.2, 8.8, 140, bg=bg)
    Ls.text(1.1, 4.8, name.upper(), "display", 0.55, ink, "mm", rotate=90)
    Ls.rect(0, 0, 2.2, 1.8, accent)
    T = Label(5.5, 2.2, 140, bg=accent)
    T.text(2.75, 1.1, name.upper(), "sans", 0.38, bg, "mm", tracking=0.2)
    Bt = Label(5.5, 2.2, 140, bg=bg)
    Bt.rect(3.6, 0.3, 5.2, 1.9, (0.85, 0.75, 0.3, 1.0))
    Bt.text(4.4, 1.1, "NYC TAX", "mono_b", 0.22, "#111111", "mm")
    for lab, rect, sc in ((F, G.FRONT, 0.1), (B, G.BACK, 0.1), (Ls, G.LEFT, 0.1), (Ls, G.RIGHT, 0.1), (T, G.TOP, 0.2), (Bt, G.BOTTOM, 0.2)):
        paste(full, age(lab.array(), seed, fade=0.05, scuff=sc), rect)
    C = Label(8.5, 2.45, 160, bg="#f7f6f2")                       # cigarette: along × around (rotated onto the island)
    C.rect(0, 0, 2.5, 2.45, filt)
    if filt != "#f0f0ea":
        C.halftone(0, 0, 2.5, 2.45, "#a8682a", 0.12, 0.035)
    C.rect(2.45, 0, 2.6, 2.45, "#c9a54a")
    C.text(3.4, 1.22, name.split()[0].upper(), "sans", 0.2, accent, "lm", tracking=0.2)
    C.rect(8.38, 0, 8.5, 2.45, "#8a8a86")
    paste(full, C.array(), G.UV_CIG, rot=1)
    return full


def logo_eagle(F):
    F.poly([(2.75, 2.3), (4.4, 3.2), (3.3, 3.3), (2.75, 4.4), (2.2, 3.3), (1.1, 3.2)], "#c9a54a")


def logo_ring(F):
    F.ellipse(2.75, 3.5, 1.1, 1.1, None, "#ffffff", 0.12); F.ellipse(2.75, 3.5, 0.6, 0.6, "#ffffff")


def logo_leaf(F):
    F.poly([(2.75, 2.2), (3.8, 3.4), (2.75, 4.6), (1.7, 3.4)], "#3aa868"); F.line([(2.75, 2.4), (2.75, 4.4)], "#ffffff", 0.05)


def logo_bars(F):
    for k in range(3):
        F.rect(1.6, 2.4 + k * 0.6, 3.9, 2.7 + k * 0.6, "#c0c6d0")


def logo_diamond(F):
    F.poly([(2.75, 2.3), (3.8, 3.4), (2.75, 4.5), (1.7, 3.4)], None or "#c9a54a"); F.poly([(2.75, 2.8), (3.3, 3.4), (2.75, 4.0), (2.2, 3.4)], "#141414")


BRANDS = {"LibertyLights": ("Liberty Lights", "SMOOTH · KING", "#f2f2ee", "#1d3f8f", "#1d3f8f", logo_eagle),
          "RedHook": ("Red Hook Reds", "FULL FLAVOR", "#c8202a", "#ffffff", "#141414", logo_ring),
          "MentholBreeze": ("Menthol Breeze", "MENTHOL · 100s", "#f2f7f2", "#1a6a3a", "#1a6a3a", logo_leaf),
          "UnionBlues": ("Union Blues", "LIGHT · KING", "#1d2b4a", "#e8ecf2", "#5a7ab0", logo_bars),
          "CanalStreet": ("Canal Street", "ULTRA · BOX", "#141414", "#c9a54a", "#c9a54a", logo_diamond)}
LIGHTERS = {"Red": (0.85, 0.10, 0.08), "Blue": (0.10, 0.30, 0.90), "Green": (0.10, 0.70, 0.20), "Yellow": (0.95, 0.80, 0.05),
            "Orange": (0.98, 0.45, 0.05), "Purple": (0.45, 0.15, 0.75), "Clear": (0.92, 0.94, 0.95), "Black": (0.06, 0.06, 0.07)}


def main():
    scene.reset()
    col = scene.collection("Game")
    P = dict(pack=G.pack(), lid=G.lid(), cig=G.cigarette(), lighter=G.lighter(), wheel=G.wheel(), button=G.button(), flame=G.flame())
    for o in P.values():
        scene.link(o, col)
    os.makedirs(WORK, exist_ok=True)
    texs = shared(np.ones((S, S), np.float32))
    foil = master.simple("M_CigPack_Foil", (0.8, 0.8, 0.82), 0.3, 1.0)
    steel = master.simple("M_Lighter_Steel", (0.62, 0.62, 0.64), 0.3, 1.0)
    flame = master.simple("M_Lighter_Flame", (1.0, 0.55, 0.15), 0.5, 0.0, emission=12.0)
    mats = {}
    for i, (k, (nm, sub, bg, ink, acc, logo)) in enumerate(BRANDS.items()):
        lp = tex.save(brand_print(nm, sub, bg, ink, acc, logo, "#f0f0ea" if k == "CanalStreet" else "#c98a3a", i),
                      os.path.join(TEXD, f"T_Cigarettes_Label_{k}.png"), "RGBA")
        prm = dict(LabelRough=0.35, WearAmount=0.35, Dirt=0.25)
        m = master.material(f"MI_CigPack_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    lighters = {}
    for k, c in LIGHTERS.items():
        g = master.glass(f"MI_Lighter_{k}", c, 0.15, 1.45)
        g.use_fake_user = True
        lighters[k] = g
    P["pack"].data.materials[0], P["pack"].data.materials[1], P["pack"].data.materials[2] = mats["LibertyLights"][0], foil, mats["LibertyLights"][0]
    for key in ("lid", "cig"):
        P[key].data.materials.clear(); P[key].data.materials.append(mats["LibertyLights"][0])
    P["lighter"].data.materials[0], P["lighter"].data.materials[1] = lighters["Red"], steel
    for key in ("wheel", "button"):
        P[key].data.materials.clear(); P[key].data.materials.append(steel if key == "wheel" else lighters["Black"])
    P["flame"].data.materials.clear(); P["flame"].data.materials.append(flame)
    tri = {o.name: scene.tris(o) for o in P.values()}
    for key, o in P.items():
        objs = [o]
        if key in ("pack", "lighter"):
            h = scene.link(export.hull(o, f"UCX_{o.name}_00", 32), scene.collection("Collision")); h.hide_render = True
            objs.append(h)
        export.fbx(objs, f"{FBXD}/{o.name}.fbx")
    layout = {"pack": (0, 0), "lid": (0, 0), "cig": (0.05, 0), "lighter": (0.09, 0), "wheel": (0.09, 0), "button": (0.09, 0), "flame": (0.09, 0)}
    for key, (x, y) in layout.items():                         # spread out in the .blend (exports stay at the origin)
        P[key].location.x += x
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "cigarettes.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        for key in ("pack", "lid"):
            d = P[key].copy(); d.data = P[key].data.copy()
            d.data.materials[0] = m
            if key == "pack":
                d.data.materials[2] = m
            d.location.x = (i - 2) * 0.075
            d.rotation_euler.z = 0.25 * (i - 2)
            d.hide_render = False
            if key == "lid" and i % 2 == 0:
                d.rotation_euler.x = math.radians(-115)
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
    for o in P.values():
        o.hide_render = True
    for j, (k, g) in enumerate(list(lighters.items())[:5]):
        for key in ("lighter", "wheel", "button"):
            d = P[key].copy(); d.data = P[key].data.copy()
            if key == "lighter":
                d.data.materials[0] = g
            d.location = ((j - 2) * 0.075 + 0.02, -0.07, 0.0)
            d.hide_render = False
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
    scene.studio(dups)
    scene.frame(dups, yaw=0, pitch=22, lens=70, fill=0.85, name="Smokes", aspect=800 / 1600)
    scene.render(f"{REND}/cigarettes_variants.png", 1600, smp, res_y=800)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "Cigarettes", "replaces_rows": ["Cigarettes", "Lighter"], "category": "smoking", "clip": "A_Consume_Smoke_Cigarette / A_Consume_Lighter_Flick",
        "dims_cm": {"SM_CigPack": [5.5, 2.2, 8.8], "SM_Cigarette": [0.78, 0.78, 8.5], "SM_Lighter": [2.5, 1.3, 6.05]},
        "pivots": {"SM_CigPack": "base centre", "SM_CigPack_Lid": "back hinge line (opens about local X to -115 deg)",
                   "SM_Cigarette": "filter end, +Z toward the tip", "SM_Lighter": "base centre", "SM_Lighter_Wheel": "axle (rotates about local X)",
                   "SM_Lighter_Button": "front edge (press -Z 1.5 mm)", "SM_Lighter_Flame": "nozzle (masked emissive card)"},
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "lighter_colours": LIGHTERS,
        "qc_notes": "20 cigarettes in the pack share one island (--uv-overlap-ok SM_CigPack); the flame card is two crossed one-sided quads"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
