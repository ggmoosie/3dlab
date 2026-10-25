"""Apple (+ 3 bitten stages): builds everything.  .venv/bin/python props/food/apple/apple.py [--fast]"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from lab import scene, master, bake, export, tex
import apple_geo as G

FAST = "--fast" in sys.argv
S = 2048
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/apple"))


def skin_art(base_hex, streak_hex, streak_amt, blush_hex=None, bruise=0.0, stripes=None, seed=0):
    """Variant print over the skin island: ground colour, vertical streaks, blush, lenticels, optional bruises."""
    u, v = tex.uv_grid(S)
    m = tex.in_rect(u, v, G.UV_SKIN)
    col = np.zeros((S, S, 3), np.float32) + np.array(tex.srgb_to_lin(tex.hex2rgb(base_hex)))
    st = tex.blur(tex.spectral(S, S, 6, 110, 1.6, seed, ax=0.04), 1.2)
    sc = np.array(tex.srgb_to_lin(tex.hex2rgb(streak_hex)))
    a = np.clip((st - (1 - streak_amt)) * 3, 0, 1)
    col = col * (1 - a[..., None]) + sc * a[..., None]
    if blush_hex:
        b = np.clip(tex.fbm(S, S, 3, seed + 1) * 1.6 - 0.5, 0, 1)
        col = col * (1 - 0.7 * b[..., None]) + np.array(tex.srgb_to_lin(tex.hex2rgb(blush_hex))) * 0.7 * b[..., None]
    if stripes:
        th = (u - G.UV_SKIN[0]) / (G.UV_SKIN[2] - G.UV_SKIN[0]) * 2 * math.pi
        sm = (np.sin(th * 7 + tex.fbm(S, S, 6, seed + 2) * 3) > 0.35).astype(np.float32)
        col = col * (1 - sm[..., None]) + np.array(tex.srgb_to_lin(tex.hex2rgb(stripes))) * sm[..., None]
    lent = tex.speckle(S, S, 0.0016, seed + 3, 1.2)
    col = col * (1 - 0.5 * lent[..., None]) + 0.5 * lent[..., None] * np.array([0.75, 0.7, 0.45])
    if bruise:
        br = np.clip((tex.fbm(S, S, 4, seed + 4) - (1 - bruise)) * 5, 0, 1)
        col = col * (1 - 0.8 * br[..., None]) + 0.8 * br[..., None] * np.array([0.10, 0.05, 0.02])
    rgba = np.concatenate([tex.lin_to_srgb(col), m[..., None].astype(np.float32)], -1)
    return rgba


def shared(ao):
    u, v = tex.uv_grid(S)
    skin, stem, flesh = tex.in_rect(u, v, G.UV_SKIN), tex.in_rect(u, v, G.UV_STEM), tex.in_rect(u, v, G.UV_FLESH)
    n1 = tex.fbm(S, S, 30, 4)
    H = np.where(skin, tex.speckle(S, S, 0.0016, 3, 1.2) * 0.04 + (tex.spectral(S, S, 40, 400, 1.2, 5) - 0.5) * 0.01, 0)
    H += np.where(flesh, (tex.spectral(S, S, 2, 60, 1.0, 6, ax=0.15) - 0.5) * 0.4 + tex.speckle(S, S, 0.004, 7, 1.0) * 0.1, 0)
    rough = np.where(skin, 0.32 + 0.1 * n1, np.where(flesh, 0.28, 0.7))
    base = np.where(skin[..., None], [0.5, 0.45, 0.1], np.where(flesh[..., None], [0.82, 0.74, 0.5], [0.12, 0.07, 0.03]))
    mR = stem.astype(np.float32); mG = flesh.astype(np.float32); mB = skin.astype(np.float32)
    wear = np.zeros((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_Apple_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.5), S / 240), p("Normal"))
    tex.save(np.stack([ao, rough, np.zeros((S, S))], -1), p("ORM"))
    tex.save(np.stack([mR, mG, mB, wear], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


STEM = (0.12, 0.07, 0.03)
VARIANTS = {
    "RedApple": (dict(base_hex="#c9b23a", streak_hex="#9e1018", streak_amt=0.8, blush_hex="#7a0a12", seed=1), (0.82, 0.74, 0.50)),
    "GreenApple": (dict(base_hex="#8fbf3a", streak_hex="#7aa830", streak_amt=0.3, seed=2), (0.86, 0.82, 0.58)),
    "GoldenApple": (dict(base_hex="#e2c24a", streak_hex="#d8a83a", streak_amt=0.3, blush_hex="#d47a2a", seed=3), (0.86, 0.80, 0.55)),
    "BruisedApple": (dict(base_hex="#b8a43a", streak_hex="#8a1418", streak_amt=0.7, bruise=0.35, seed=4), (0.68, 0.55, 0.30)),
    "Banapple": (dict(base_hex="#efd94a", streak_hex="#e8c83a", streak_amt=0.2, stripes="#6aa832", seed=5), (0.92, 0.88, 0.62)),
}


def main():
    scene.reset()
    col = scene.collection("Game")
    flesh = bpy.data.materials.new("MI_Apple_Flesh")
    a0 = scene.link(G.apple(), col)
    stages = [scene.link(G.bitten(k, flesh), col) for k in (1, 2, 3)]
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (S, 32)
    ao = bake.bake(a0, [], "AO", n, f"{WORK}/ao.png", samples=smp)
    if n != S:
        from PIL import Image
        ao = np.asarray(Image.fromarray((np.clip(ao, 0, 1) * 255).astype(np.uint8)).resize((S, S)), np.float32) / 255
    texs = shared(np.where(ao < 0.02, 1.0, ao))
    mats = {}
    for k, (art, flesh_col) in VARIANTS.items():
        lp = tex.save(skin_art(**art), os.path.join(TEXD, f"T_Apple_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=STEM, RoughA=0.7, MetalA=0.0, ColorB=flesh_col, RoughB=0.25, MetalB=0.0, LabelRough=0.34, WearAmount=0.0, Dirt=0.2)
        m = master.material(f"MI_Apple_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in [a0] + stages:
        for i in range(len(o.data.materials)):
            o.data.materials[i] = mats["RedApple"][0]
    tri = {o.name: scene.tris(o) for o in [a0] + stages}
    for o in [a0] + stages:
        h = scene.link(export.hull(o, f"UCX_{o.name}_00", 64), scene.collection("Collision"))
        h.hide_render = True
        export.fbx([o, h], f"{FBXD}/{o.name}.fbx")
    for i, o in enumerate(stages):                             # alternates, not an assembly: spread them in the .blend
        o.location.x = (i + 1) * 0.1
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "apple.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    for i, o in enumerate([a0] + stages):                      # bite sequence, left to right
        o.location = ((i - 1.5) * 0.09, 0, 0.0375)
        o.rotation_euler.z = math.radians(180 - 25)
    scene.studio([a0] + stages)
    scene.frame([a0] + stages, yaw=0, pitch=14, lens=70, fill=0.85, name="Bites", aspect=600 / 1600)
    scene.render(f"{REND}/apple_bites.png", 1600, smp, res_y=600)
    for o in stages:
        o.hide_render = True
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        d = a0.copy(); d.data = a0.data.copy()
        for j in range(len(d.data.materials)):
            d.data.materials[j] = m
        d.location = ((i - 2) * 0.085, 0, 0.0375); d.rotation_euler.z = i * 1.3
        bpy.context.scene.collection.objects.link(d)
        dups.append(d)
    a0.hide_render = True
    for c in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(c)
    scene.frame(dups, yaw=0, pitch=18, lens=70, fill=0.85, name="Variants", aspect=600 / 1600)
    scene.render(f"{REND}/apple_variants.png", 1600, smp, res_y=600)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "Apple", "replaces_rows": ["Apple", "Banapple"], "category": "food", "clip": "A_Consume_Eat_Apple",
        "dims_cm": [7.7, 7.7, 7.85], "real_object": "medium apple (Ø77 mm, 70 mm + stem)", "pivot": "centre",
        "parts": {f"SM_Apple_Bite{k}": {"swap_on": f"Bite{k}"} for k in (1, 2, 3)},
        "material_slots": ["MI_Apple_<V> (skin; stem = layer A)", "MI_Apple_<V> (flesh = layer B on the bite faces)"],
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "collision": "UCX_SM_Apple*_00 hulls"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
