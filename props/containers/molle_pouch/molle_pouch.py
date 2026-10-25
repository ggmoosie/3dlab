"""GPPouch (MOLLE general-purpose pouch + flap): builds everything.  .venv/bin/python props/containers/molle_pouch/molle_pouch.py [--fast]
Texture detail is placed in 3D via a position bake: weave (triplanar), PALS stitching + 38 mm bartacks, panel seams, velcro."""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from PIL import Image
from lab import scene, master, bake, export, tex
import pouch_geo as G

FAST = "--fast" in sys.argv
S = 2048
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/molle_pouch"))


def up(a):
    if a.shape[0] == S:
        return a
    if a.ndim == 2:
        return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((S, S)), np.float32)
    return np.stack([up(a[..., i]) for i in range(a.shape[-1])], -1)


def detail(pos, nrm, ao, which):
    """pos/nrm: (S,S,3) world mm / unit normals per texel. Returns height (mm), roughness, wear, hardware mask."""
    x, y, z = pos[..., 0] * 1000, pos[..., 1] * 1000, pos[..., 2] * 1000
    ax = np.argmax(np.abs(nrm), -1)
    a = np.where(ax == 0, y, x); b = np.where(ax == 2, y, z)              # triplanar coordinates
    weave = (np.abs(np.sin(a * math.pi / 0.55)) ** 0.6 + np.abs(np.sin(b * math.pi / 0.55)) ** 0.6) * 0.5
    H = weave * 0.05
    stitch = np.zeros_like(x)
    def line(d, dash_coord, width=0.35):                                 # a stitch row at distance d=0
        return np.exp(-(d / width) ** 2) * (np.sin(dash_coord * math.pi / 1.6) > -0.2)
    if which == "body":
        front = nrm[..., 1] < -0.5
        for zc in G.PALS_Z:                                              # double stitch along each webbing edge + bartacks
            for e in (zc - 11.0, zc + 11.0):
                stitch += front * line(z - e, x)
            for k in range(-2, 3):
                bx = k * 38.0
                stitch += front * (np.abs(x - bx) < 1.5) * (np.abs(z - zc) < 10) * (np.sin(z * math.pi / 0.7) > 0)
        edge = np.minimum.reduce([G.HX - np.abs(x), G.HY - np.abs(y), np.abs(z - 2 * G.HZ), z]) if False else None
        seam_d = np.minimum(np.abs(np.abs(x) - (G.HX - 3.0)), np.abs(z - 4.0))
        stitch += line(seam_d, np.where(np.abs(np.abs(x) - (G.HX - 3.0)) < 1, z, x)) * (np.abs(nrm[..., 1]) > 0.5)
    if which == "flap":
        fx = np.abs(x) - (G.HX - 5.0)
        stitch += line(np.minimum(np.abs(fx), 99), z)
        velcro = (np.abs(x) < 40) & (nrm[..., 2] > 0.6)
        H += velcro * (tex.speckle(*x.shape, 0.08, 9, 0.7) * 0.25)
    H += stitch * 0.25
    rough = 0.82 - 0.1 * weave
    wear = np.clip((1 - ao) * 0.0 + tex.fbm(*x.shape, 40, 5) * 0, 0, 1)
    return H, rough, stitch


def shared(P):
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (S, 24)
    H = np.zeros((S, S), np.float32); rough = np.full((S, S), 0.8, np.float32); ao = np.ones((S, S), np.float32)
    stitchmap = np.zeros((S, S), np.float32)
    hw = np.zeros((S, S), np.float32)
    u, v = tex.uv_grid(S)
    hw = (tex.in_rect(u, v, (0.64, 0.14, 0.99, 0.28)) | tex.in_rect(u, v, (0.64, 0.0, 0.99, 0.13)) & (u > 0.80)).astype(np.float32)
    for key, o in P.items():
        a = up(bake.bake(o, [], "AO", n, f"{WORK}/ao_{key}.png", samples=smp))
        pos, msk = bake.bake(o, [], "POS", n, f"{WORK}/pos_{key}.png", samples=1)
        nrm = bake.bake(o, [], "ONRM", n, f"{WORK}/nrm_{key}.png", samples=1)
        pos, msk, nrm = up(pos), up(msk.astype(np.float32)) > 0.5, up(nrm)
        h, r, st = detail(pos, nrm, a, "body" if key == "pouch" else "flap")
        H = np.where(msk, h, H); rough = np.where(msk, r, rough); ao = np.where(msk & (a > 0.02), a, ao)
        stitchmap = np.where(msk, st, stitchmap)
    base = np.zeros((S, S, 3), np.float32) + [0.45, 0.45, 0.45]
    base = np.where(hw[..., None] > 0, [0.05, 0.05, 0.05], base)
    one = np.ones((S, S), np.float32)
    fabric = 1 - hw
    wear = np.clip(tex.scratches(S, S, 160, (0.005, 0.03), 1, 6) * 0.5 + (1 - ao) * 0.0, 0, 1) * fabric
    p = lambda k: os.path.join(TEXD, f"T_GPPouch_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.4), S / 300), p("Normal"))
    tex.save(np.stack([ao * (0.8 + 0.2 * (1 - stitchmap.clip(0, 1))), np.where(hw > 0, 0.45, rough), 0 * one], -1), p("ORM"))
    tex.save(np.stack([0 * one, hw, fabric, wear], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}, stitchmap


def camo(cols, scale, seed, digital=False):
    """A camouflage print over the whole UV space (blob layers; 'digital' quantises to 6 px squares)."""
    c = [np.array(tex.srgb_to_lin(tex.hex2rgb(h))) for h in cols]
    out = np.zeros((S, S, 3), np.float32) + c[0]
    for k, col in enumerate(c[1:]):
        f = tex.spectral(S, S, scale, scale * 6, 2.6, seed + k)
        if digital:
            q = 5
            f = np.repeat(np.repeat(f[::q, ::q], q, 0), q, 1)[:S, :S]
        out = np.where((f > 0.55 + 0.04 * k)[..., None], col, out)
    return out


def solid(hexc, seed):
    c = np.array(tex.srgb_to_lin(tex.hex2rgb(hexc)))
    return np.zeros((S, S, 3), np.float32) + c * (0.94 + 0.12 * tex.fbm(S, S, 8, seed)[..., None])


VARIANTS = {"Coyote": (lambda: solid("#8a7350", 1), (0.03, 0.03, 0.03)),
            "RangerGreen": (lambda: solid("#4a5038", 2), (0.03, 0.03, 0.03)),
            "Black": (lambda: solid("#1c1c1e", 3), (0.02, 0.02, 0.02)),
            "WolfGrey": (lambda: solid("#5a5c5e", 4), (0.03, 0.03, 0.03)),
            "Woodland": (lambda: camo(["#6a6a48", "#3a4a2a", "#5a4a32", "#1c1c16"], 5, 10), (0.25, 0.20, 0.12)),
            "UrbanDigital": (lambda: camo(["#7a7c7e", "#55575b", "#3a3c40", "#9a9c9e"], 10, 20, True), (0.03, 0.03, 0.03))}


def main():
    scene.reset()
    col = scene.collection("Game")
    P = {"pouch": scene.link(G.pouch(), col), "flap": scene.link(G.flap(), col)}
    texs, stitch = shared(P)
    mats = {}
    for k, (fn, hwcol) in VARIANTS.items():
        art = fn()
        art = art * (1 - 0.35 * stitch.clip(0, 1)[..., None]) + 0.35 * stitch.clip(0, 1)[..., None] * art * 0.6   # thread a shade darker
        rgba = np.concatenate([tex.lin_to_srgb(art), np.ones((S, S, 1), np.float32)], -1)
        lp = tex.save(rgba, os.path.join(TEXD, f"T_GPPouch_Label_{k}.png"), "RGBA")
        prm = dict(ColorB=hwcol, RoughB=0.45, MetalB=0.0, LabelRough=0.85, WearAmount=0.4, Dirt=0.45)
        m = master.material(f"MI_GPPouch_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in P.values():
        o.data.materials[0] = mats["Coyote"][0]
    lods = [export.lod(P["pouch"], r, f"SM_GPPouch_LOD{i + 1}") for i, r in enumerate((0.5, 0.25))]
    for o in lods:
        o.hide_render = True; o.hide_set(True)
    tri = {o.name: scene.tris(o) for o in list(P.values()) + lods}
    h = scene.link(export.hull(P["pouch"], "UCX_SM_GPPouch_00", 48), scene.collection("Collision")); h.hide_render = True
    export.fbx([P["pouch"], h], f"{FBXD}/SM_GPPouch.fbx")
    export.fbx([P["flap"]], f"{FBXD}/SM_GPPouch_Flap.fbx")
    for o in lods:
        export.fbx([o], f"{FBXD}/{o.name}.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "molle_pouch.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        for o in P.values():
            d = o.copy(); d.data = o.data.copy(); d.data.materials[0] = m
            d.location.x += (i - 2.5) * 0.18
            d.rotation_euler.z = 0.3 * (i - 2.5) / 2.5
            if o is P["flap"] and i == 0:
                d.rotation_euler.x = math.radians(-100)
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
    for o in P.values():
        o.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=0, pitch=14, lens=70, fill=0.88, name="Pouches", aspect=700 / 1600)
    scene.render(f"{REND}/molle_pouch_variants.png", 1600, smp, res_y=700)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "GPPouch", "replaces_rows": ["SmallContainer"], "category": "containers", "clip": None,
        "dims_cm": [15.1, 8.3, 17.1], "real_object": "MOLLE general-purpose pouch, 500D nylon, 3 PALS rows", "pivot": "base centre",
        "parts": {"SM_GPPouch_Flap": {"pivot": "rear hinge at the top", "opens": "about local X to -100 deg"}},
        "grid_suggestion": "2x2 container, 2x3 inside",
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "collision": "UCX_SM_GPPouch_00",
        "lods": ["SM_GPPouch_LOD1 (50%)", "SM_GPPouch_LOD2 (25%)"]})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
