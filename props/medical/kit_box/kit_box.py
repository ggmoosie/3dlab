"""KitBox (first-aid / surgery tin + lid + scalpel): builds everything.  .venv/bin/python props/medical/kit_box/kit_box.py [--fast]"""
import os, sys, math, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from PIL import Image
from lab import scene, master, bake, export, tex
from lab.labels import Label, age
import kit_geo as G

FAST = "--fast" in sys.argv
S = 2048
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/kit_box"))
FOAM = (0.50, 0.25, 0.99, 0.48)
PACKS = [(0.50, 0.0, 0.73, 0.23), (0.75, 0.0, 0.98, 0.23)]


def up(a):
    return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((S, S)), np.float32) / 255


def shared(ao, curv):
    u, v = tex.uv_grid(S)
    paint = tex.in_rect(u, v, G.UV_TRAY) | tex.in_rect(u, v, G.UV_REST) | tex.in_rect(u, v, G.UV_LIDTOP)
    foam = tex.in_rect(u, v, FOAM)
    packs = tex.in_rect(u, v, PACKS[0]) | tex.in_rect(u, v, PACKS[1])
    n1, n2 = tex.fbm(S, S, 10, 1), tex.fbm(S, S, 60, 2)
    base = np.where(paint[..., None], [0.58, 0.58, 0.6], np.where(foam[..., None], [0.05, 0.05, 0.055], [0.85, 0.85, 0.83]))
    iod = tex.in_rect(u, v, (0.734, 0.0, 0.747, 0.23))                               # amber iodine bottle
    base = np.where(iod[..., None], [0.22, 0.07, 0.015], base)
    metal = paint.astype(np.float32)
    rough = np.where(paint, 0.35, np.where(foam, 0.95, np.where(iod, 0.18, 0.5))).astype(np.float32)
    H = np.where(foam, (tex.fbm(S, S, 120, 3) - 0.5) * 0.4, 0) + np.where(packs, (tex.spectral(S, S, 4, 40, 2.2, 4) - 0.5) * 0.3, 0)
    H += np.where(paint, (tex.fbm(S, S, 6, 5) - 0.5) * 0.15, 0)                       # soft dents in the tin
    edge = np.clip((curv - 0.55) * 6, 0, 1)
    wear = np.where(paint, np.clip(edge * (0.4 + n2) + tex.scratches(S, S, 120, (0.005, 0.04), 1, 6) * 0.8 * (n1 > 0.5), 0, 1), 0)
    one = np.ones((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_KitBox_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(H, 0.5), S / 260), p("Normal"))
    tex.save(np.stack([ao, rough, metal], -1), p("ORM"))
    mB = (tex.in_rect(u, v, G.UV_LIDTOP) | packs).astype(np.float32)
    tex.save(np.stack([paint.astype(np.float32), 0 * one, mB, wear], -1), p("Mask"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}


def packet(title, sub, color):
    P = Label(6.4, 4.8, 120, bg="#f4f4f0")
    P.rect(0, 0, 6.4, 0.7, color)
    P.text(3.2, 0.35, "STERILE", "sans", 0.3, "#ffffff", "mm", tracking=0.3)
    P.text(3.2, 1.6, title, "heavy", 0.42, "#1a1a1a", "mm", max_w=5.8)
    P.text(3.2, 2.35, sub, "sans_r", 0.24, "#1a1a1a", "mm", max_w=5.8)
    P.text(3.2, 3.6, "Do not use if package is damaged · LOT 4471", "sans_r", 0.15, "#1a1a1a", "mm")
    P.ellipse(5.6, 4.2, 0.35, 0.35, None, color, 0.04)
    return P


def lid_art(kind):
    W, H = 16.16, 10.16
    L = Label(W, H, 120)
    if kind == "FirstAid":
        L.rect(W / 2 - 0.9, H / 2 - 2.9, W / 2 + 0.9, H / 2 + 2.9, "#ffffff"); L.rect(W / 2 - 2.9, H / 2 - 0.9, W / 2 + 2.9, H / 2 + 0.9, "#ffffff")
        L.text(W / 2, H - 1.3, "FIRST AID", "heavy", 0.75, "#ffffff", "mm", tracking=0.12)
    elif kind == "Surgical":
        L.rect(0.8, 0.8, W - 0.8, H - 0.8, None, outline="#1d3f8f", width=0.06)
        L.text(W / 2, H / 2 - 0.6, "SURGICAL KIT", "sans", 0.85, "#1d3f8f", "mm", tracking=0.2)
        L.text(W / 2, H / 2 + 0.6, "STERILE INSTRUMENTS  ·  SINGLE PATIENT USE", "sans_r", 0.3, "#1d3f8f", "mm", tracking=0.08)
    elif kind == "Military":
        L.text(W / 2, H / 2 - 1.1, "KIT, FIRST AID", "stencil", 0.85, "#16180f", "mm", tracking=0.1)
        L.text(W / 2, H / 2 + 0.2, "INDIVIDUAL · GENERAL PURPOSE", "stencil", 0.45, "#16180f", "mm", tracking=0.08)
        L.text(W / 2, H / 2 + 1.4, "NSN 6545-01-532-3674", "mono_b", 0.42, "#16180f", "mm")
    elif kind == "CookieTin":
        L.gradient(0, 0, W, H, "#1c4fa0", "#163c7a")
        for k in range(3):
            for j in range(2):
                cx, cy = 3.4 + k * 4.7, 4.3 + j * 3.6
                L.ellipse(cx, cy, 1.6, 1.6, "#d9a35a"); L.ellipse(cx, cy, 1.2, 1.2, "#e8bc72")
                for a in range(12):
                    L.ellipse(cx + 1.4 * math.cos(a * 0.52), cy + 1.4 * math.sin(a * 0.52), 0.12, 0.12, "#b8823a")
        L.text(W / 2, 1.3, "HOLMBERG", "serif_b", 0.85, "#f2e6c8", "mm", tracking=0.1)
        L.text(W / 2, 2.25, "DANISH-STYLE BUTTER COOKIES", "serif", 0.32, "#f2e6c8", "mm", tracking=0.12)
        L.rect(W / 2 - 3.2, H / 2 - 0.9, W / 2 + 3.2, H / 2 + 0.9, (0.93, 0.89, 0.72, 0.95))          # masking tape
        L.text(W / 2, H / 2, "MEDS — NOT COOKIES", "hand", 0.7, "#1a1a1a", "mm", rotate=-2)
    elif kind == "Vintage":
        L.rect(0, 0, W, H, "#e9e1c8")
        L.ellipse(W / 2, H / 2 - 0.4, 2.6, 2.6, "#1f7a74")
        L.rect(W / 2 - 0.6, H / 2 - 2.2, W / 2 + 0.6, H / 2 + 1.4, "#e9e1c8"); L.rect(W / 2 - 2.2, H / 2 - 1.0, W / 2 + 2.2, H / 2 + 0.2, "#e9e1c8")
        L.text(W / 2, H - 1.5, "JOHNSTOWN FIRST AID CO.", "serif_b", 0.5, "#1f7a74", "mm", tracking=0.06)
        L.text(W / 2, 1.1, "HOUSEHOLD PACKET · EST. 1911", "serif", 0.3, "#1f7a74", "mm", tracking=0.1)
    return L


VARIANTS = {"FirstAid": ((0.55, 0.03, 0.03), "#c8202a"), "Surgical": ((0.80, 0.80, 0.78), "#1d3f8f"),
            "Military": ((0.13, 0.15, 0.08), "#3a4a22"), "CookieTin": ((0.08, 0.20, 0.50), "#1c4fa0"),
            "Vintage": ((0.75, 0.70, 0.55), "#1f7a74")}


def compose(kind, color):
    full = np.zeros((S, S, 4), np.float32)
    parts = [(lid_art(kind), G.UV_LIDTOP, 0.12), (packet("GAUZE PADS", "4 × 4 in · 12-ply · 10 pads", color), PACKS[0], 0.05),
             (packet("TRAUMA DRESSING", "6 in emergency bandage", color), PACKS[1], 0.05)]
    for i, (lab, rect, sc) in enumerate(parts):
        art = age(lab.array(), i + len(kind), fade=0.05, scuff=sc)
        u0, v0, u1, v1 = rect
        j0, j1, i0, i1 = int(u0 * S), int(u1 * S), int((1 - v1) * S), int((1 - v0) * S)
        full[i0:i1, j0:j1] = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0), Image.LANCZOS), np.float32) / 255
    return full


def main():
    scene.reset()
    col = scene.collection("Game")
    box, lid, sc = scene.link(G.tray(), col), scene.link(G.lid(), col), scene.link(G.scalpel(), col)
    from mathutils import Matrix
    sc.matrix_world = Matrix.Translation((0, -28 * G.MM, 3.5 * G.MM)) @ Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (S, 24)
    ao_b = bake.bake(box, [], "AO", n, f"{WORK}/ao_box.png", samples=smp)
    ao_l = bake.bake(lid, [], "AO", n, f"{WORK}/ao_lid.png", samples=smp)
    def dense(o):                                    # pointiness needs a dense mesh: bake curvature from a subdivided copy
        h = o.copy(); h.data = o.data.copy(); h.name = o.name + "_curv"
        bpy.context.scene.collection.objects.link(h)
        m = h.modifiers.new("Sub", "SUBSURF"); m.levels = m.render_levels = 2; m.subdivision_type = "SIMPLE"
        return h
    hb, hl_ = dense(box), dense(lid)
    cv_b = bake.bake(box, [hb], "CURV", n, f"{WORK}/cv_box.png", samples=4, extrusion=0.0005, max_ray=0.002)
    cv_l = bake.bake(lid, [hl_], "CURV", n, f"{WORK}/cv_lid.png", samples=4, extrusion=0.0005, max_ray=0.002)
    for h in (hb, hl_):
        bpy.data.objects.remove(h)
    if n != S:
        ao_b, ao_l, cv_b, cv_l = map(up, (ao_b, ao_l, cv_b, cv_l))
    u, v = tex.uv_grid(S)
    lidm = tex.in_rect(u, v, G.UV_REST) | tex.in_rect(u, v, G.UV_LIDTOP)
    ao = np.where(lidm, ao_l, ao_b)
    texs = shared(np.where(ao < 0.02, 1.0, ao), np.where(lidm, cv_l, cv_b))
    steel = master.simple("M_Scalpel_Steel", (0.62, 0.62, 0.64), 0.22, 1.0)
    mats = {}
    for k, (paint, accent) in VARIANTS.items():
        lp = tex.save(compose(k, accent), os.path.join(TEXD, f"T_KitBox_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=paint, RoughA=0.4, MetalA=0.15, LabelRough=0.45, WearAmount=0.5, Dirt=0.35)
        m = master.material(f"MI_KitBox_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in (box, lid):
        o.data.materials.clear(); o.data.materials.append(mats["FirstAid"][0])
    sc.data.materials.clear(); sc.data.materials.append(steel)
    tri = {o.name: scene.tris(o) for o in (box, lid, sc)}
    h = scene.link(export.hull(box, "UCX_SM_KitBox_00", 32), scene.collection("Collision")); h.hide_render = True
    export.fbx([box, h], f"{FBXD}/SM_KitBox.fbx")
    export.fbx([lid], f"{FBXD}/SM_KitBox_Lid.fbx")
    mw = sc.matrix_world.copy()
    sc.matrix_world = Matrix()
    export.fbx([sc], f"{FBXD}/SM_Scalpel.fbx")
    sc.matrix_world = mw
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "kit_box.blend"), relative_remap=True)
    smp = 16 if FAST else 96
    dups = []
    for i, (k, (m, _, _)) in enumerate(mats.items()):
        for o in (box, lid) + ((sc,) if i == 0 else ()):
            d = o.copy(); d.data = o.data.copy()
            if o is not sc:
                d.data.materials[0] = m
            d.location.x += (i - 2) * 0.19
            d.location.z += G.BH / 2 * G.MM
            if o is lid:
                d.rotation_euler.x = math.radians(-110) if i % 2 == 0 else 0.0
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
    for o in (box, lid, sc):
        o.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=0, pitch=40, lens=70, fill=0.85, name="Kits", aspect=700 / 1600)
    scene.render(f"{REND}/kit_box_variants.png", 1600, smp, res_y=700)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "KitBox", "replaces_rows": ["FirstAidTin", "MedKitLarge"], "category": "medical", "clip": "A_Consume_Surgery_Kit",
        "dims_cm": [16.0, 10.0, 5.0], "real_object": "hinged steel first-aid tin", "pivot": "centre",
        "parts": {"SM_KitBox_Lid": {"pivot": "hinge line on the far long edge (+Y Blender / -Y UE)", "opens": "about local X to -110 deg"},
                  "SM_Scalpel": {"pivot": "pinch point, 4 cm from the tail", "axis": "+Z toward the blade", "dims_cm": [0.8, 0.22, 13.2]}},
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "collision": "UCX_SM_KitBox_00"})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
