"""Spoon: builds everything.  .venv/bin/python props/food/spoon/spoon.py [--fast]
Variants: stainless, tan ration-pack plastic, black plastic (layer A = the whole spoon)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from lab import scene, master, bake, export, tex
import spoon_geo as G

FAST = "--fast" in sys.argv
S = 1024
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/spoon"))
VARIANTS = {"Steel": dict(ColorA=(0.62, 0.60, 0.57), RoughA=0.16, MetalA=1.0, WearAmount=0.0, Dirt=0.25),
            "RationTan": dict(ColorA=(0.42, 0.33, 0.21), RoughA=0.48, MetalA=0.0, WearAmount=0.0, Dirt=0.3),
            "BlackPlastic": dict(ColorA=(0.025, 0.025, 0.027), RoughA=0.42, MetalA=0.0, WearAmount=0.0, Dirt=0.3)}


def textures(o):
    n, smp = (512, 8) if FAST else (S, 32)
    ao = bake.bake(o, [], "AO", n, f"{WORK}/ao.png", samples=smp)
    if n != S:
        from PIL import Image
        ao = np.asarray(Image.fromarray((np.clip(ao, 0, 1) * 255).astype(np.uint8)).resize((S, S)), np.float32) / 255
    ao = np.where(ao < 0.02, 1.0, ao)
    scr = tex.scratches(S, S, 500, (0.01, 0.06), 1, 5)
    hmap = scr * 0.004 + (tex.fbm(S, S, 20, 6) - 0.5) * 0.002
    rough = 0.18 + 0.12 * scr + 0.04 * tex.fbm(S, S, 8, 7)
    p = lambda k: os.path.join(TEXD, f"T_Spoon_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(np.zeros((S, S, 3)) + 0.62), p("BaseColor"))
    tex.save(tex.height_to_normal(hmap, S / 60.0), p("Normal"))
    tex.save(np.stack([ao, rough, np.ones((S, S))], -1), p("ORM"))
    one = np.ones((S, S), np.float32)
    tex.save(np.stack([one, 0 * one, 0 * one, scr * 0.3], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


def main():
    scene.reset()
    o = scene.link(G.spoon(), scene.collection("Game"))
    os.makedirs(WORK, exist_ok=True)
    texs = textures(o)
    mats = {}
    for k, prm in VARIANTS.items():
        m = master.material(f"MI_Spoon_{k}", texs, prm)
        m.use_fake_user = True
        mats[k] = m
    o.data.materials.append(mats["Steel"])
    ucx = scene.link(export.hull(o, "UCX_SM_Spoon_00", 48), scene.collection("Collision"))
    ucx.hide_render = True
    tri = scene.tris(o)
    export.fbx([o, ucx], f"{FBXD}/SM_Spoon.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "spoon.blend"), relative_remap=True)
    # renders: three spoons lying side by side, bowls up
    smp = 16 if FAST else 96
    dups = []
    for i, k in enumerate(VARIANTS):
        d = o.copy(); d.data = o.data.copy()
        d.data.materials[0] = mats[k]
        d.rotation_euler = (0, -1.5708, 0.12 * (i - 1))
        d.location = (-0.03 + 0.0, (i - 1) * 0.038, 0.0075)
        bpy.context.scene.collection.objects.link(d)
        dups.append(d)
    o.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=-25, pitch=48, lens=70, fill=0.85, name="Spoons")
    scene.render(f"{REND}/spoon.png", 1024, smp)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "Spoon", "replaces_rows": [], "category": "food", "clip": "A_Consume_Eat_CanSpoon (right hand)",
        "dims_cm": [1.39, 3.02, 17.0], "real_object": "dessert spoon, 170 mm", "pivot": "pinch point 5 cm from the handle end; +Z toward the bowl; +X = concave side",
        "tris": {"SM_Spoon": tri}, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: None for k in VARIANTS}},
        "variant_params": VARIANTS, "collision": "UCX_SM_Spoon_00"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
