"""FlatBar (flow-wrapped bar, tear-off end): builds everything.  .venv/bin/python props/food/flat_bar/flat_bar.py [--fast]
The wrapper sheet is 11.64 × 12.8 cm unrolled (perimeter × length); u=0/1 is the back fin seam, u=0.5 the front centre."""
import os, sys, math, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from PIL import Image
from lab import scene, master, bake, export, tex
from lab.labels import Label, age
import bar_geo as G

FAST = "--fast" in sys.argv
S = 2048
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/flat_bar"))
_, PER = G.section()
SW, SL = PER / 10, G.L / 10                     # sheet cm
FX = SW / 2                                     # front centre (cm)


def shared(ao):
    u, v = tex.uv_grid(S)
    sheet, barm = tex.in_rect(u, v, G.UV_SHEET), tex.in_rect(u, v, G.UV_BAR)
    z = (v - G.UV_SHEET[1]) / (G.UV_SHEET[3] - G.UV_SHEET[1]) * G.L - G.L / 2
    fin = (np.abs(z) > G.L / 2 - G.FIN)
    crimp = (np.abs(((u * 140) % 1.0) - 0.5) * 2) * fin
    H = np.where(sheet, crimp * 0.15 + (tex.spectral(S, S, 3, 40, 2.2, 3, ax=0.6) - 0.5) * 0.12, 0)
    bu = (u - G.UV_BAR[0]) / (G.UV_BAR[2] - G.UV_BAR[0])
    bv = (v - G.UV_BAR[1]) / (G.UV_BAR[3] - G.UV_BAR[1])
    seg = np.minimum(np.abs(((bu * 3) % 1) - 0.5), np.abs(((bv * 8) % 1) - 0.5))
    H += np.where(barm, np.clip(seg * 4, 0, 1) * 0.6, 0)
    rough = np.where(sheet, 0.3, 0.45).astype(np.float32)
    base = np.where(sheet[..., None], [0.8, 0.8, 0.8], [0.12, 0.06, 0.03])
    metal = np.where(sheet, 0.0, 0.0)
    wear = np.where(sheet, np.clip(fin * tex.fbm(S, S, 30, 4) - 0.3, 0, 1), 0)
    one = np.ones((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_FlatBar_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.5), S / 116.4), p("Normal"))
    tex.save(np.stack([ao, rough, metal], -1), p("ORM"))
    tex.save(np.stack([0 * one, 0 * one, one, wear], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


def product(choc, speck=None, speck_amt=0.0, seed=0):
    """Variant print for the bar island: base colour + inclusions (nuts, oats, crisp)."""
    P = Label(5.0, 10.4, 80, bg=choc)
    if speck:
        rng = np.random.default_rng(seed)
        for _ in range(int(400 * speck_amt)):
            x, y = rng.uniform(0, 5, 2) * (1, 2.08)
            c = speck[int(rng.integers(len(speck)))]
            P.ellipse(x, y, rng.uniform(0.03, 0.12), rng.uniform(0.02, 0.08), c)
    return P


def wrapper(bg, title, sub, ink, accent, kcal, extra=None, font="display"):
    L = Label(SW, SL, S / SW, bg=bg)
    if extra:
        extra(L)
    L.text(FX, SL / 2 - 0.4, title, font, 1.15, ink, "mm", tracking=0.04, max_w=SW * 0.42)
    L.text(FX, SL / 2 + 0.75, sub, "sans", 0.3, accent, "mm", tracking=0.18, max_w=SW * 0.42)
    L.text(FX, SL - 1.8, f"NET WT 1.55 OZ (44g)  ·  {kcal} kcal", "sans", 0.2, ink, "mm", tracking=0.08)
    for k in (0, 1):                                      # back panel either side of the fin seam
        bx = SW * (0.12 if k == 0 else 0.88)
        L.text(bx, SL / 2, "INGREDIENTS: SUGAR, COCOA BUTTER, MILK, COCOA MASS, SOY LECITHIN, VANILLA.", "sans_r", 0.14, ink, "mm",
               rotate=90)
    L.barcode(SW * 0.2, SL - 2.6, 1.6, 0.8, "0" + str(zlib.crc32(title.encode()) % 10 ** 11).zfill(11))
    return L


def stripes(colors):
    def f(L):
        for i, c in enumerate(colors):
            L.rect(0, SL * (0.12 + i * 0.03), SW, SL * (0.12 + i * 0.03 + 0.018), c)
            L.rect(0, SL * (0.82 - i * 0.03), SW, SL * (0.82 - i * 0.03 + 0.018), c)
    return f


VARIANTS = {
    "BrooklynMilk": (lambda: wrapper("#4b1f6b", "BROOKLYN", "MILK CHOCOLATE", "#f2d27a", "#ffffff", 230, stripes(["#f2d27a", "#ffffff"])),
                     lambda: product("#5a2d16", None), 0.25),
    "BrooklynDark": (lambda: wrapper("#151313", "BROOKLYN", "DARK 72% CACAO", "#d9b26a", "#d9b26a", 220, stripes(["#d9b26a"])),
                     lambda: product("#2e160b", None), 0.4),
    "BrooklynAlmond": (lambda: wrapper("#1e4a8f", "BROOKLYN", "MILK · ALMONDS", "#f2d27a", "#ffffff", 240, stripes(["#f2d27a"])),
                       lambda: product("#5a2d16", ["#c9a06a", "#b8895a"], 0.8, 2), 0.25),
    "TrailMix": (lambda: wrapper("#c8a678", "Trail Mix", "OATS · HONEY · RAISIN", "#2f5a2a", "#7a3a12", 190, None, "serif_i"),
                 lambda: product("#b88a4a", ["#e0c08a", "#5a2a1a", "#8a5a2a"], 1.6, 3), 0.5),
    "Max30": (lambda: wrapper("#121212", "MAX 30g", "PROTEIN · CHOC CRISP", "#ff7a1a", "#ffffff", 290, stripes(["#ff7a1a"]), "heavy"),
              lambda: product("#4a2a18", ["#d9c09a", "#3a1e10"], 1.2, 4), 0.5),
    "Survival400": (lambda: wrapper("#c9cbcd", "SURVIVAL 400", "EMERGENCY FOOD RATION", "#141414", "#a01010", 400, stripes(["#a01010"]), "stencil"),
                    lambda: product("#d8c7a0", ["#c9b68a"], 0.6, 5), 0.7),
}


def to_uv(L, P, seed):
    full = np.zeros((S, S, 4), np.float32)
    for lab, rect, sc in ((L, G.UV_SHEET, 0.08), (P, G.UV_BAR, 0.0)):
        art = age(lab.array(), seed, fade=0.04, scuff=sc) if sc else lab.array()
        u0, v0, u1, v1 = rect
        j0, j1, i0, i1 = int(round(u0 * S)), int(round(u1 * S)), int(round((1 - v1) * S)), int(round((1 - v0) * S))
        full[i0:i1, j0:j1] = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0), Image.LANCZOS), np.float32) / 255
    return full


def main():
    scene.reset()
    col = scene.collection("Game")
    a, t = scene.link(G.flat_bar(), col), scene.link(G.tear_end(), col)
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (S, 24)
    ao = bake.bake(a, [], "AO", n, f"{WORK}/ao.png", samples=smp)
    if n != S:
        ao = np.asarray(Image.fromarray((np.clip(ao, 0, 1) * 255).astype(np.uint8)).resize((S, S)), np.float32) / 255
    texs = shared(np.where(ao < 0.02, 1.0, ao))
    foil = master.simple("M_FlatBar_Foil", (0.8, 0.8, 0.82), 0.25, 1.0)
    mats = {}
    for k, (wf, pf, lr) in VARIANTS.items():
        lp = tex.save(to_uv(wf(), pf(), len(k)), os.path.join(TEXD, f"T_FlatBar_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=(0.5, 0.5, 0.5), RoughA=0.4, MetalA=0.0, ColorB=(0.5, 0.5, 0.5), RoughB=0.4, MetalB=0.0,
                   LabelRough=lr, WearAmount=0.3, Dirt=0.25)
        m = master.material(f"MI_FlatBar_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in (a, t):
        o.data.materials[0], o.data.materials[1] = mats["BrooklynMilk"][0], foil
    tri = {o.name: scene.tris(o) for o in (a, t)}
    h = scene.link(export.hull(a, "UCX_SM_FlatBar_00", 32), scene.collection("Collision")); h.hide_render = True
    export.fbx([a, h], f"{FBXD}/SM_FlatBar.fbx")
    export.fbx([t], f"{FBXD}/SM_FlatBar_TearEnd.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "flat_bar.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    t.location.z += 0.012; t.rotation_euler = (0.3, 0.2, 0.15)       # tear-off end, pulled away
    for o in (a, t):
        o.rotation_euler.x -= math.radians(90)
    bpy.context.view_layer.update()
    scene.studio([a, t])
    scene.frame([a, t], yaw=-20, pitch=40, lens=85, fill=0.85, name="Hero")
    scene.render(f"{REND}/flat_bar.png", 1024, smp)
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        d = a.copy(); d.data = a.data.copy()
        d.data.materials[0] = m
        d.location = ((i - 2.5) * 0.064, 0, 0.003)
        d.rotation_euler = (math.radians(-90), 0, 0)
        bpy.context.scene.collection.objects.link(d)
        dups.append(d)
    a.hide_render = t.hide_render = True
    for c in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(c)
    scene.frame(dups, yaw=0, pitch=62, lens=70, fill=0.85, name="Variants", aspect=800 / 1600)
    scene.render(f"{REND}/flat_bar_variants.png", 1600, smp, res_y=800)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "FlatBar", "replaces_rows": ["Chocolate"], "category": "food", "clip": "A_Consume_Eat_BarFlat",
        "dims_cm": [5.42, 0.44, 12.8], "real_object": "44 g flow-wrapped bar", "pivot": "centre, +Z = bite end",
        "parts": {"SM_FlatBar_TearEnd": {"pivot": "on the tear line (z 4.6 cm)", "opens": "pull off along +Z, twist ~20 deg"}},
        "material_slots": ["MI_FlatBar_<V> (wrapper print + bar)", "M_FlatBar_Foil (film inside)"],
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "collision": "UCX_SM_FlatBar_00",
        "qc_notes": "film inner faces reuse the outer UVs in the foil slot (--uv-overlap-ok); the bar rides loose inside (--allow-float SM_FlatBar)"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
