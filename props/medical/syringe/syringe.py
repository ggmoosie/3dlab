"""Syringe (5 mL, needle cap, plunger, liquid): builds everything.  .venv/bin/python props/medical/syringe/syringe.py [--fast]"""
import os, sys, math, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from PIL import Image
from lab import scene, master, bake, export, tex
from lab.labels import Label, age
import syringe_geo as G

FAST = "--fast" in sys.argv
S = 1024
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/syringe"))
LW, LH = (G.RO + 0.15) * math.radians(150) / 10, (G.Z1 - G.Z0 - 12) / 10      # label cm (around × along)


def shared(ao):
    u, v = tex.uv_grid(S)
    cap, hub, plg, needle = (tex.in_rect(u, v, r) for r in (G.UV_CAP, G.UV_HUB, G.UV_PLUNGER, (0.27, 0.0, 0.48, 0.015)))
    stopper = tex.in_rect(u, v, (0.72, 0.64, 0.99, 0.99))
    base = np.zeros((S, S, 3), np.float32) + 0.85
    base = np.where(needle[..., None], 0.75, base)
    metal = needle.astype(np.float32)
    rough = np.where(needle, 0.2, np.where(stopper, 0.6, 0.35)).astype(np.float32)
    H = np.where(cap, (np.cos(u * 2 * math.pi * 120) * 0.5 + 0.5) * 0.06, 0)          # cap grip ribs
    one = np.ones((S, S), np.float32)
    mR = (cap | hub).astype(np.float32)
    mG = stopper.astype(np.float32)
    mB = tex.in_rect(u, v, G.UV_LABEL).astype(np.float32)
    p = lambda k: os.path.join(TEXD, f"T_Syringe_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(H, S / 60), p("Normal"))
    tex.save(np.stack([ao, rough, metal], -1), p("ORM"))
    tex.save(np.stack([mR, mG, mB, 0 * one], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


def drug_label(name, dose, band, ink="#141414", dark=False, seed=0):
    """Laid out landscape (along the barrel × around), then turned 90° onto the label island."""
    W, H = LH, LW
    L = Label(W, H, 220, bg="#141414" if dark else "#f7f7f4")
    L.rect(0, 0, 1.1, H, band)
    L.text(0.55, H / 2, name[:4], "heavy", 0.34, "#141414", "mm", rotate=90)
    fg = "#f2f2f2" if dark else ink
    L.text(1.3, 0.35, name, "heavy", 0.36, fg, "lm", tracking=0.04, max_w=W - 1.5)
    L.text(1.3, 0.8, dose, "sans", 0.22, fg, "lm", max_w=W - 1.5)
    L.text(1.3, 1.15, f"Rx only · single dose · LOT {zlib.crc32(name.encode()) % 90000 + 10000} · EXP 03/27", "sans_r", 0.13, fg, "lm", max_w=W - 1.5)
    for k in range(11):                                       # graduations 0–5 mL along the bottom edge
        x = W - 0.25 - k * (W - 1.6) / 10
        L.rect(x, H - 0.3 - (0.15 if k % 2 == 0 else 0), x + 0.025, H - 0.08, fg)
        if k % 2 == 0:
            L.text(x, H - 0.55, str(k // 2), "sans", 0.13, fg, "mm")
    art = age(L.array(), seed, fade=0.04, scuff=0.08)
    art = np.rot90(art, 1)
    full = np.zeros((S, S, 4), np.float32)
    u0, v0, u1, v1 = G.UV_LABEL
    j0, j1, i0, i1 = int(u0 * S), int(u1 * S), int((1 - v1) * S), int((1 - v0) * S)
    full[i0:i1, j0:j1] = np.asarray(Image.fromarray((np.ascontiguousarray(art) * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0), Image.LANCZOS), np.float32) / 255
    return full


VARIANTS = {   # name, dose, band, cap/hub colour, liquid tint, dark label
    "EpiShot": ("EPISHOT", "epinephrine 1 mg/mL", "#f2c21a", (0.85, 0.62, 0.02), (0.98, 0.95, 0.75), False),
    "Morphex": ("MORPHEX", "morphine 10 mg/mL", "#c8202a", (0.60, 0.03, 0.04), (0.97, 0.97, 0.97), False),
    "Analgex": ("ANALGEX", "ketorolac 30 mg/mL", "#2a62c8", (0.05, 0.18, 0.60), (0.80, 0.88, 1.0), False),
    "Atrox": ("ATROX", "atropine 2 mg/0.7 mL", "#2a9a4a", (0.05, 0.40, 0.12), (0.95, 0.98, 0.95), False),
    "Rush": ("RUSH", "combat stimulant · 1 dose", "#ff7a1a", (0.95, 0.35, 0.02), (1.0, 0.55, 0.12), True),
    "Amoxiflex": ("AMOXIFLEX", "amoxicillin 500 mg", "#8a4ab0", (0.85, 0.85, 0.85), (0.92, 0.90, 0.85), False),
}


def main():
    scene.reset()
    col = scene.collection("Game")
    P = dict(barrel=G.barrel(), label=G.label(), plunger=G.plunger(), cap=G.cap(), liquid=G.liquid())
    for o in P.values():
        scene.link(o, col)
    os.makedirs(WORK, exist_ok=True)
    n, smp = (512, 8) if FAST else (S, 24)
    ao = np.ones((S, S), np.float32)
    texs = shared(ao)
    clear = master.glass("M_Syringe_Clear", (0.97, 0.98, 0.99), 0.08, 1.45, normal=texs["Normal"])
    mats = {}
    for i, (k, (nm, dose, band, capc, liq, dark)) in enumerate(VARIANTS.items()):
        lp = tex.save(drug_label(nm, dose, band, dark=dark, seed=i), os.path.join(TEXD, f"T_Syringe_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=capc, RoughA=0.4, MetalA=0.0, ColorB=(0.02, 0.02, 0.02), RoughB=0.55, MetalB=0.0, LabelRough=0.6, WearAmount=0.0, Dirt=0.2)
        m = master.material(f"MI_Syringe_{k}", texs, prm, label=lp)
        lq = master.glass(f"MI_Syringe_Liquid_{k}", liq, 0.0, 1.34)
        for x in (m, lq):
            x.use_fake_user = True
        mats[k] = dict(mi=m, liq=lq, params=prm, label=lp, liquid=liq)

    def dress(Q, k):
        m = mats[k]
        Q["barrel"].data.materials[0], Q["barrel"].data.materials[1] = clear, m["mi"]
        for key in ("label", "plunger", "cap"):
            me = Q[key].data
            if not me.materials:
                me.materials.append(m["mi"])
            me.materials[0] = m["mi"]
        if not Q["liquid"].data.materials:
            Q["liquid"].data.materials.append(m["liq"])
        Q["liquid"].data.materials[0] = m["liq"]
    dress(P, "EpiShot")
    tri = {o.name: scene.tris(o) for o in P.values()}
    h = scene.link(export.hull(P["barrel"], "UCX_SM_Syringe_00", 48), scene.collection("Collision")); h.hide_render = True
    export.fbx([P["barrel"], P["label"], h], f"{FBXD}/SM_Syringe.fbx")
    for key in ("plunger", "cap", "liquid"):
        export.fbx([P[key]], f"{FBXD}/{P[key].name}.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "syringe.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    s = bpy.context.scene
    s.cycles.transmission_bounces = s.cycles.max_bounces = 12
    dups = []
    for i, k in enumerate(VARIANTS):
        Q = {}
        for key, o in P.items():
            d = o.copy(); d.data = o.data.copy()
            bpy.context.scene.collection.objects.link(d)
            Q[key] = d
        dress(Q, k)
        if i == 0:                                           # first one: cap pulled
            Q["cap"].location.z += 0.045
        bpy.context.view_layer.update()
        for d in Q.values():
            d.matrix_world = __import__("mathutils").Matrix.Translation(((i - 2.5) * 0.042, 0, 0.0095)) @ \
                __import__("mathutils").Matrix.Rotation(math.radians(90), 4, "X") @ __import__("mathutils").Matrix.Rotation(math.pi, 4, "Z") @ d.matrix_world
        dups += Q.values()
    for o in P.values():
        o.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=-90 + 90, pitch=60, lens=70, fill=0.85, name="Syringes", aspect=800 / 1600)
    scene.render(f"{REND}/syringe_variants.png", 1600, smp, res_y=800)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "Syringe", "replaces_rows": ["AdrenalineSyringe", "MorphineSyringe", "PainkillerSyringe"], "category": "medical",
        "clip": "A_Consume_Inject_Forearm / A_Consume_Syringe_Thigh", "dims_cm": [3.4, 1.3, 12.0], "real_object": "5 mL luer syringe + 23G needle",
        "pivot": "barrel centre, +Z toward the needle",
        "parts": {"SM_Syringe_Plunger": {"pivot": "thumb pad centre", "travel_cm": 3.8, "axis": "+Z"},
                  "SM_Syringe_Cap": {"pivot": "open end", "pull_cm": 4.5, "axis": "+Z"},
                  "SM_Syringe_Liquid": {"pivot": "front of the bore", "note": "scale Z with the plunger travel"}},
        "material_slots": {"SM_Syringe": ["M_Lab_Glass (barrel)", "MI_Syringe_<V> (hub, needle, label)"]},
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v["label"]) for k, v in mats.items()}},
        "variant_params": {k: v["params"] for k, v in mats.items()}, "liquid_tints": {k: v["liquid"] for k, v in mats.items()},
        "collision": "UCX_SM_Syringe_00", "qc_notes": "label is a declared decal (qc_open)"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
