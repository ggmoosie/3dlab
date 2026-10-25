""".50-cal style steel ammo can: body, hinged lid with rubber gasket, cam latch, folding carry handle. Builds everything.
    .venv/bin/python props/ammo/ammo_can/ammo_can.py [--fast]
Parts (moving parts are separate meshes, pivot on their joint):
  SM_AmmoCan          body (pressed-steel box, hinge leaves, latch bracket), pivot bottom centre, long axis +Y
  SM_AmmoCan_Lid      lid shell + gasket + hinge knuckles + handle lugs, pivot on the hinge axis (+X side), opens +105 deg about +Y
  SM_AmmoCan_Latch    cam latch lever on the -Y end, pivot on its pin, opens +100 deg about +X (swings out and down)
  SM_AmmoCan_Handle   carry handle, pivot on the lug axis (lid top centre line), lifts 90 deg about +Y
Skins (mask material): OD, Tan, Rusty. Real object: M2A1-pattern .50 cal can, 30.5 x 15.5 x 19 cm."""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, os.path.dirname(HERE))
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector
from lab import scene, master, bake, export, tex, geo
from lab.labels import Label, age
import box_lib

FAST = "--fast" in sys.argv
MM = 0.001
S = 2048
TEXD, FBXD, REND, WORK = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders", "../../../work/ammo_can"))
BW, BL, BH = 140.0, 275.0, 175.0            # body
LW, LL, LT, LS = 146.0, 281.0, 182.0, 22.0  # lid outer W, L, top z, skirt depth
ST = 1.2                                    # sheet steel
HX, HZ = LW / 2 + 3.5, 168.0                # hinge axis (x, z)
LATCH_Z = 118.0                             # latch pin height on the -Y end
HANDLE_Z = LT + 8.0                         # handle pivot height
LY = -BL / 2 - 7.0                          # latch pin (y)


def shell_box(name, w, l, h, z0, r, open_side="top", t=ST):
    """Rounded-corner box w x l x h from z0, one face open, made solid with sheet thickness t (inward)."""
    o = geo.rounded_box(name, w, l, h, r, z0, 0.0, segs=4)
    bm = bmesh.new(); bm.from_mesh(o.data)
    sel = [f for f in bm.faces if (f.normal.z > 0.9 and open_side in ("top", "both")) or (f.normal.z < -0.9 and open_side in ("bottom", "both"))]
    bmesh.ops.delete(bm, geom=sel, context="FACES")
    hor = [e for e in bm.edges if len(e.link_faces) == 2 and abs(e.verts[0].co.z - e.verts[1].co.z) < 1e-7]
    bmesh.ops.bevel(bm, geom=hor, offset=2.0 * MM, segments=2, affect="EDGES", profile=0.5)
    bm.to_mesh(o.data); bm.free()
    s = o.modifiers.new("S", "SOLIDIFY"); s.thickness = t * MM; s.offset = -1.0; s.use_even_offset = True
    geo.apply_mods(o)
    return o


def cyl(name, r, l, loc, axis="Y", segs=12):
    bpy.ops.mesh.primitive_cylinder_add(vertices=segs, radius=r * MM, depth=l * MM, location=[c * MM for c in loc])
    o = bpy.context.object
    o.name = name
    if axis == "Y":
        o.rotation_euler.x = math.pi / 2
    elif axis == "X":
        o.rotation_euler.y = math.pi / 2
    geo.bake_xform(o)
    return o


def box(name, w, l, h, x, y, z0, r=0.0):
    return geo.rounded_box(name, w, l, h, r, z0, 0.0, segs=2, center=(x, y))


def tidy(o):
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4], quad_method="BEAUTY")
    bm.to_mesh(o.data); bm.free()
    return o


def body():
    b = shell_box("SM_AmmoCan", BW, BL, BH, 0.0, 9.0)
    # stacking bead round the waist and a rolled top rim (proud rings fused into the shell)
    for z, hh, d in ((60.0, 5.0, 1.4), (BH - 6.0, 6.0, 1.0)):
        ring = shell_box("ring", BW + 2 * d, BL + 2 * d, hh, z, 9.0 + d, open_side="both", t=d + 0.6)
        geo.boolean(b, ring, "UNION")
    # hinge leaves on +X, latch bracket on -Y
    for y in (-80.0, 80.0):
        leaf = box("leaf", 6.0, 22.0, 18.0, BW / 2 + 2.5, y - 11.0, 140.0)
        geo.boolean(b, leaf, "UNION")
        up_ = box("up", 2.4, 20.0, 13.0, HX - 1.5, y - 11.0, 155.0)
        geo.boolean(b, up_, "UNION")
        k = cyl("kn", 3.2, 20.0, (HX, y - 11.0, HZ))
        geo.boolean(b, k, "UNION")
    br = box("br", 52.0, 6.0, 16.0, 0.0, -BL / 2 - 2.5, LATCH_Z - 8.0)
    geo.boolean(b, br, "UNION")
    return tidy(b)


def lid():
    l = shell_box("SM_AmmoCan_Lid", LW, LL, LS, LT - LS, 11.0, open_side="bottom")
    # pressed panel on the top (raised 1.2 mm), gasket ring under the top, hinge knuckles on +X, handle lugs on the top
    pan = box("pan", LW - 34, LL - 50, 1.6, 0, 0, LT - 0.4, 6.0)
    geo.boolean(l, pan, "UNION")
    gk = geo.rounded_box("gk", BW + 1.0, BL + 1.0, LT - ST + 0.3 - (BH + 0.3), 9.5, BH + 0.3, 0.0, segs=4)
    cut = geo.rounded_box("gc", BW - 12, BL - 12, 30, 4.0, BH - 5, 0.0, segs=4)
    geo.boolean(gk, cut)
    geo.boolean(l, gk, "UNION")
    for y in (-80.0, 80.0):
        arm = box("arm", 7.6, 18.0, 6.0, LW / 2 + 2.6, y + 11.0, HZ - 2.0)
        geo.boolean(l, arm, "UNION")
        k = cyl("kn", 3.2, 18.0, (HX, y + 11.0, HZ))
        geo.boolean(l, k, "UNION")
    for y in (-96.0, 96.0):
        lug = box("lug", 12.0, 5.0, 9.0, 0.0, y, LT + 0.5)
        geo.boolean(l, lug, "UNION")
    # catch lip on the -Y skirt for the latch hook
    lip = box("lip", 40.0, 3.0, 3.8, 0.0, -LL / 2 - 1.2, LT - 10.5)
    geo.boolean(l, lip, "UNION")
    return tidy(l)


def latch():
    """Lever plate with a top hook over the lid lip, pin boss at the bottom."""
    y0 = LY - 0.5
    p = box("SM_AmmoCan_Latch", 46.0, 3.0, LT - 6.0 - LATCH_Z, 0.0, y0 - 1.5, LATCH_Z, 1.0)
    hook = box("hook", 36.0, 6.0, 3.0, 0.0, y0 + 0.5, LT - 7.0 - 3.0 + 0.0)
    geo.boolean(p, hook, "UNION")
    boss = cyl("boss", 4.0, 58.0, (0.0, y0 + 0.5, LATCH_Z + 2.0), axis="X", segs=12)
    geo.boolean(p, boss, "UNION")
    tab = box("tab", 30.0, 8.0, 2.4, 0.0, y0 - 5.0, LT - 8.8)
    geo.boolean(p, tab, "UNION")
    return tidy(p)


def handle():
    """U strap lying flat toward +X: arms from the lug pins to a rolled grip along Y."""
    h = cyl("SM_AmmoCan_Handle", 3.0, 196.0, (36.0, 0.0, HANDLE_Z), axis="Y", segs=10)
    grip = cyl("grip", 6.0, 120.0, (36.0, 0.0, HANDLE_Z), axis="Y", segs=12)
    geo.boolean(h, grip, "UNION")
    for y in (-98.0, 98.0):
        arm = box("arm", 40.0, 4.0, 5.0, 18.0, y, HANDLE_Z - 2.5)
        geo.boolean(h, arm, "UNION")
        pin = cyl("pin", 2.6, 14.0, (0.0, y - (5 if y > 0 else -5), HANDLE_Z), axis="Y", segs=10)
        geo.boolean(h, pin, "UNION")
    return tidy(h)


def stencil(ink, seed, faded=False):
    def side(w, h):
        L = Label(w, h, 40)
        L.text(w / 2, h * 0.27, "CAL .50", "stencil", h * 0.2, ink, "mm", tracking=0.08)
        L.text(w / 2, h * 0.47, "AMMUNITION · 100 CARTRIDGES", "stencil", h * 0.07, ink, "mm", max_w=w * 0.85)
        L.text(w / 2, h * 0.56, "PATRIOT ARMS CO. · LOT PA-26-0118", "stencil", h * 0.05, ink, "mm", max_w=w * 0.85)
        L.rect(w * 0.12, h * 0.8, w * 0.88, h * 0.82, ink)
        return L
    def top(w, h):
        L = Label(w, h, 40)
        L.text(w / 2, h * 0.5, "PATRIOT ARMS CO.", "stencil", h * 0.16, ink, "mm", max_w=w * 0.7)
        return L
    def end(w, h):
        L = Label(w, h, 40)
        L.text(w / 2, h * 0.3, ".50", "stencil", h * 0.18, ink, "mm")
        L.text(w / 2, h * 0.5, "LINKED", "stencil", h * 0.09, ink, "mm")
        return L
    return side, top, end


VARIANTS = {
    "OD": dict(ColorA=(0.075, 0.085, 0.045), RoughA=0.55, MetalA=0.0, ink="#d8c23a", WearAmount=0.3, Dirt=0.5),
    "Tan": dict(ColorA=(0.36, 0.28, 0.16), RoughA=0.6, MetalA=0.0, ink="#1a1a16", WearAmount=0.3, Dirt=0.5),
    "Rusty": dict(ColorA=(0.07, 0.075, 0.045), RoughA=0.7, MetalA=0.0, ink="#bfa944", WearAmount=0.88, Dirt=1.0),
}


def main():
    scene.reset()
    col, colc = scene.collection("Game"), scene.collection("Collision")
    bo, li, la, ha = (scene.link(f(), col) for f in (body, lid, latch, handle))
    # ---- UVs: body sides dieline-mapped (stencil), everything else smart-projected into its own rect
    R = {"side_a": (0.0, 0.0, 0.48, 0.30), "side_b": (0.5, 0.0, 0.98, 0.30), "end_a": (0.0, 0.32, 0.25, 0.48),
         "end_b": (0.27, 0.32, 0.52, 0.48), "lid_top": (0.54, 0.32, 0.98, 0.56), "body_misc": (0.0, 0.5, 0.5, 0.99),
         "lid_misc": (0.52, 0.58, 0.99, 0.99), "latch": (0.0, 0.0, 0.0, 0.0), "handle": (0.0, 0.0, 0.0, 0.0)}
    R["body_misc"] = (0.0, 0.5, 0.45, 0.95)
    R["lid_misc"] = (0.47, 0.58, 0.92, 0.99)
    R["latch"] = (0.94, 0.58, 0.999, 0.75)
    R["handle"] = (0.94, 0.77, 0.999, 0.99)
    zr = (8.0, 165.0)
    box_lib.axis_uv(bo, {"-X": [(R["side_a"], -BW / 2, (-BL / 2 + 9, BL / 2 - 9), zr)],
                         "+X": [(R["side_b"], BW / 2, (-BL / 2 + 9, BL / 2 - 9), zr)],
                         "-Y": [(R["end_a"], -BL / 2, (-BW / 2 + 9, BW / 2 - 9), zr)],
                         "+Y": [(R["end_b"], BL / 2, (-BW / 2 + 9, BW / 2 - 9), zr)]}, R["body_misc"], tol=0.4)
    box_lib.axis_uv(li, {"+Z": [(R["lid_top"], LT + 1.2, (-LW / 2 + 17, LW / 2 - 17), (-LL / 2 + 25, LL / 2 - 25))]}, R["lid_misc"], tol=0.5)
    geo.unwrap_into(la, R["latch"])
    geo.unwrap_into(ha, R["handle"])
    for o in (bo, li, la, ha):
        geo.fix_zero_uv(o, (0.94, 0.5, 0.999, 0.56))
        geo.weighted_normals(o)
    geo.set_origin(li, (HX * MM, 0, HZ * MM))
    geo.set_origin(la, (0, LY * MM, (LATCH_Z + 2.0) * MM))
    geo.set_origin(ha, (0, 0, HANDLE_Z * MM))
    # ---- bakes: AO per part (assembled), convexity from simple-subdivided copies (pointiness needs density)
    os.makedirs(WORK, exist_ok=True)
    n, smp = (1024, 8) if FAST else (S, 24)
    aos, cvs = {}, {}
    for o in (bo, li, la, ha):
        aos[o.name] = bake.bake(o, [], "AO", n, f"{WORK}/ao_{o.name}.png", samples=smp)
        hi = o.copy(); hi.data = o.data.copy(); bpy.context.scene.collection.objects.link(hi)
        for m in list(hi.modifiers):
            hi.modifiers.remove(m)
        sd = hi.modifiers.new("Sub", "SUBSURF"); sd.subdivision_type = "SIMPLE"; sd.levels = sd.render_levels = 3
        cvs[o.name] = bake.bake(o, [hi], "CURV", n, f"{WORK}/cv_{o.name}.png", samples=1, extrusion=0.0015, max_ray=0.004)
        bpy.data.objects.remove(hi)
    from PIL import Image
    up = lambda a: np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8)).resize((S, S)), np.float32) / 255 if a.shape[0] != S else a
    u, v = tex.uv_grid(S)
    own = {bo.name: ["side_a", "side_b", "end_a", "end_b", "body_misc"], li.name: ["lid_top", "lid_misc"], la.name: ["latch"], ha.name: ["handle"]}
    ao = np.ones((S, S), np.float32)
    cv = np.full((S, S), 0.5, np.float32)
    for nm, keys in own.items():
        m = np.zeros((S, S), bool)
        for k in keys:
            m |= tex.in_rect(u, v, R[k])
        ao = np.where(m, up(aos[nm]), ao)
        cv = np.where(m, up(cvs[nm]), cv)
    ao = np.where(ao < 0.02, 1.0, ao)
    cv = np.where(cv < 0.01, 0.5, cv)
    # ---- substrate (steel with rust), masks
    n1, n2, n3 = tex.fbm(S, S, 8, 1), tex.fbm(S, S, 30, 2), tex.fbm(S, S, 90, 3)
    rust = np.clip((n1 * 0.7 + n2 * 0.5 - 0.45) * 2.2, 0, 1)
    steel = np.array((0.30, 0.30, 0.30), np.float32)
    rustc = np.array((0.28, 0.11, 0.035), np.float32) * (0.7 + 0.6 * n3[..., None])
    base = steel * (1 - rust[..., None]) + rustc * rust[..., None]
    rough = (0.45 + 0.45 * rust).astype(np.float32)
    metal = (1 - rust * 0.9).astype(np.float32)
    # gasket: lid faces below the lid top (inside, z < LT - ST) that sit inside the body outline → texel-space via the
    # lid-misc island is not known per face, so mark it geometrically: give gasket faces material-free ID by position
    bpy.context.view_layer.update()
    gas = gasket_mask(li, R["lid_misc"])
    edges = tex.remap(cv, 0.55, 0.75)
    wear = np.clip(edges * (0.55 + 0.6 * n2) + tex.remap(n1 * 0.6 + n3 * 0.5, 0.45, 0.95) * 0.9, 0, 1)
    wear = np.maximum(wear, tex.scratches(S, S, 700, (0.004, 0.03), 1, 7) * 0.8)
    mR = np.where(gas, 0.0, 1.0).astype(np.float32)
    mG = gas.astype(np.float32)
    pm = np.zeros((S, S), bool)
    for k in ("side_a", "side_b", "end_a", "end_b", "lid_top"):
        pm |= tex.in_rect(u, v, R[k])
    hgt = (n3 - 0.5) * 0.02 + tex.scratches(S, S, 500, (0.004, 0.02), 1, 9) * -0.05
    for k in ("side_a", "side_b"):                                          # pressed stiffening panel on the long sides
        r = R[k]
        a = (u - r[0]) / (r[2] - r[0]); b = (v - r[1]) / (r[3] - r[1])
        inner = (a > 0.08) & (a < 0.92) & (b > 0.42) & (b < 0.80)
        hgt += np.where(tex.in_rect(u, v, r), tex.blur(inner.astype(np.float32), 3) * 0.6, 0)
    one = np.ones((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_AmmoCan_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(tex.blur(hgt, 0.7), 3.0), p("Normal"))
    tex.save(np.stack([ao, rough, metal], -1), p("ORM"))
    tex.save(np.stack([mR, mG, pm.astype(np.float32), wear], -1), p("Mask"), "RGBA")
    texs = {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
    mats = {}
    for i, (k, vp) in enumerate(VARIANTS.items()):
        side, top, end = stencil(vp["ink"], i)
        full = np.zeros((S, S, 4), np.float32)
        sw, sh = (BL - 18) / 10, (zr[1] - zr[0]) / 10
        box_lib.paste(full, side(sw, sh).array(), R["side_a"])
        box_lib.paste(full, side(sw, sh).array(), R["side_b"])
        ew = (BW - 18) / 10
        box_lib.paste(full, end(ew, sh).array(), R["end_a"])
        box_lib.paste(full, end(ew, sh).array(), R["end_b"])
        box_lib.paste(full, np.rot90(top((LL - 50) / 10, (LW - 34) / 10).array(), -1), R["lid_top"])
        full = age(full, i + 3, fade=0.1, scuff=0.6 if k == "Rusty" else 0.3)
        if k == "Rusty":
            full[..., 3] *= 0.75
        lp = tex.save(full, os.path.join(TEXD, f"T_AmmoCan_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=vp["ColorA"], RoughA=vp["RoughA"], MetalA=vp["MetalA"], ColorB=(0.02, 0.02, 0.02), RoughB=0.85, MetalB=0.0,
                   LabelRough=0.6, WearAmount=vp["WearAmount"], Dirt=vp["Dirt"])
        m = master.material(f"MI_AmmoCan_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
    for o in (bo, li, la, ha):
        if o.data.materials:
            o.data.materials[0] = mats["OD"][0]
        else:
            o.data.materials.append(mats["OD"][0])
    # ---- collision, LODs, export
    hb = scene.link(export.hull(bo, "UCX_SM_AmmoCan_00", 32), colc); hb.hide_render = True
    hl = scene.link(export.hull(li, "UCX_SM_AmmoCan_Lid_00", 32), colc); hl.hide_render = True
    parts = (bo, li, la, ha)
    tri = {o.name: scene.tris(o) for o in parts}
    lods = []
    if tri[bo.name] + tri[li.name] > 2000:
        for o in (bo, li):
            for r_, s_ in ((0.5, "LOD1"), (0.25, "LOD2")):
                d = export.lod(o, r_, f"{o.name}_{s_}")
                d.hide_render = True; d.hide_set(True)
                for c in list(d.users_collection):
                    c.objects.unlink(d)
                scene.collection("LODs").objects.link(d)
                tri[d.name] = scene.tris(d)
                lods.append(d)
    export.fbx([bo, hb], f"{FBXD}/SM_AmmoCan.fbx")
    export.fbx([li, hl], f"{FBXD}/SM_AmmoCan_Lid.fbx")
    export.fbx([la], f"{FBXD}/SM_AmmoCan_Latch.fbx")
    export.fbx([ha], f"{FBXD}/SM_AmmoCan_Handle.fbx")
    for d in lods:
        export.fbx([d], f"{FBXD}/{d.name}.fbx")
        d.hide_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "ammo_can.blend"), relative_remap=True)
    renders(parts, mats)
    lo, hi_ = scene.bounds(list(parts))
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "AmmoCan", "replaces_rows": ["AmmoCan (container, new)"], "category": "ammo", "clip": None,
        "real_object": "M2A1-pattern .50 cal steel ammunition can, 30.5 x 15.5 x 19 cm",
        "dims_cm": [round((hi_[i] - lo[i]) * 100, 2) for i in range(3)], "pivot": "bottom centre, long axis +Y, latch end -Y",
        "parts": {"SM_AmmoCan_Lid": {"pivot": "hinge axis (+X side)", "rest_location_cm": [HX / 10, 0, HZ / 10], "rest_rotation_deg": [0, 0, 0],
                                     "opens": "about local +Y, 0 -> 105 deg (latch must be open first)"},
                  "SM_AmmoCan_Latch": {"pivot": "latch pin on the -Y end", "rest_location_cm": [0, LY / 10, (LATCH_Z + 2) / 10],
                                       "rest_rotation_deg": [0, 0, 0], "opens": "about local +X, 0 -> +100 deg (swings out, -Y)"},
                  "SM_AmmoCan_Handle": {"pivot": "lug axis on the lid top (child of the lid)", "rest_location_cm": [0, 0, HANDLE_Z / 10],
                                        "rest_rotation_deg": [0, 0, 0], "opens": "about local +Y, 0 -> -90 deg (lifts upright)", "parent": "SM_AmmoCan_Lid"}},
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "mask": "R paint · G rubber gasket · B stencil · A chips / rust-through",
        "collision": ["UCX_SM_AmmoCan_00", "UCX_SM_AmmoCan_Lid_00"], "lods": [d.name for d in lods]})
    print("TRIS", tri)


def gasket_mask(li, rect):
    """Texel mask of the gasket faces: rasterise the UV triangles of lid faces lying inside the gasket band."""
    from PIL import Image, ImageDraw
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    me = li.data
    uv = me.uv_layers.active.data
    for p in me.polygons:
        c = (li.matrix_world @ p.center) / MM
        if not (BH + 0.1 < c.z < LT - ST - 0.05):
            continue
        inside = abs(c.x) < BW / 2 + 1.0 and abs(c.y) < BL / 2 + 1.0
        if not inside:
            continue
        pts = [(uv[i].uv.x * S, (1 - uv[i].uv.y) * S) for i in p.loop_indices]
        d.polygon(pts, fill=255)
    return np.asarray(im, np.float32) > 127


def place(parts, M, lid_deg=0, latch_deg=0, handle_deg=0, mat=None):
    bo, li, la, ha = parts
    out = {}
    for o in parts:
        d = o.copy(); d.data = o.data.copy()
        d.hide_render = False
        if mat:
            d.data.materials[0] = mat
        bpy.context.scene.collection.objects.link(d)
        out[o.name] = d
    bpy.context.view_layer.update()
    L = Matrix.Translation(li.location) @ Matrix.Rotation(math.radians(lid_deg), 4, "Y") @ Matrix.Translation(-li.location)
    out[li.name].matrix_world = M @ L @ li.matrix_world
    out[ha.name].matrix_world = M @ L @ Matrix.Translation(ha.location) @ Matrix.Rotation(math.radians(handle_deg), 4, "Y")
    out[la.name].matrix_world = M @ Matrix.Translation(la.location) @ Matrix.Rotation(math.radians(latch_deg), 4, "X")
    out[bo.name].matrix_world = M @ bo.matrix_world
    return list(out.values())


def renders(parts, mats):
    for o in list(bpy.data.collections["Game"].objects) + list(bpy.data.collections["Collision"].objects):
        o.hide_render = True
    smp = 16 if FAST else 48
    objs = []
    for i, k in enumerate(mats):
        M = Matrix.Translation(((i - 1) * 0.36, 0, 0)) @ Matrix.Rotation(math.radians(90), 4, "Z")
        objs += place(parts, M, mat=mats[k][0], handle_deg=-90 if i == 1 else 0)
    scene.studio(objs)
    scene.frame(objs, yaw=-22, pitch=22, lens=60, fill=0.92, name="Skins", aspect=576 / 1024)
    scene.render(os.path.join(REND, "ammo_can.png"), 1024, smp, res_y=576)
    for o in objs:
        bpy.data.objects.remove(o)
    for o in [o for o in bpy.data.objects if o.type in ("LIGHT", "CAMERA") or o.name.startswith("Backdrop")]:
        bpy.data.objects.remove(o)
    # rest vs open (latch down, lid up, handle up) for the moving-part gate
    objs = place(parts, Matrix.Translation((-0.2, 0, 0)) @ Matrix.Rotation(math.radians(90), 4, "Z"), mat=mats["OD"][0])
    objs += place(parts, Matrix.Translation((0.2, 0, 0)) @ Matrix.Rotation(math.radians(90), 4, "Z"), 105, 100, -90, mats["OD"][0])
    scene.studio(objs)
    scene.frame(objs, yaw=-28, pitch=26, lens=60, fill=0.9, name="Open", aspect=576 / 1024)
    scene.render(os.path.join(REND, "ammo_can_open.png"), 1024, smp, res_y=576)


if __name__ == "__main__":
    main()
