"""Folded-cardboard ammo boxes (tuck-top) with a pull-out tray, one per ammo type. Shared builder (bpy + numpy + PIL).
Parts (each its own mesh, pivot at its joint):
  SM_<Item>        box body: bottom + 4 walls, 0.6 mm board, pivot bottom centre, long axis +Y, print faces outward
  SM_<Item>_Lid    lid panel + tuck flap, pivot on the hinge line (top of the +X wall), opens +115 deg about local +Y
  SM_<Item>_Tray   plastic / board tray, pivot bottom centre, slides up out of the box (+Z)
One texture set per box (the three parts share UVs); one material instance. Rounds in the tray are not part of the
box mesh: the manifest lists the slot positions (cm, box space) so the game can instance SM_<Round>_Low in them;
the hero render does exactly that.
Used by props/ammo/box_<cal>/box_<cal>.py:  box_lib.build("9mm", HERE)"""
import os, sys, math, json, zlib
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
sys.path.insert(0, ROOT)
import bpy, bmesh
import numpy as np
from PIL import Image
from mathutils import Matrix, Vector
from lab import scene, master, bake, export, tex, geo
from lab.labels import Label, age

FAST = "--fast" in sys.argv
MM = 0.001
T = 0.6                  # board thickness (mm)
TL = 0.6                 # lid board
AMMO = os.path.dirname(os.path.abspath(__file__))

BOXES = {
    "9mm": dict(item="AmmoBox9mm", rows=["9mmRoundBox"], W=74, D=128, H=34, nx=5, ny=10, count=50,
                round=("round_9mm", "Round9mm_Low", "FMJ"), cal="9x19 mm", cal_big="9 mm", sub="124 GR FMJ",
                kind="CENTERFIRE PISTOL CARTRIDGES", color="#1d4f9e", accent="#f2c230", ink="#ffffff", tray=(0.85, 0.85, 0.83),
                real="50-round 9 mm range box (tuck top, white tray), sized to the DT_Ammo row 12.8 x 7.4 x 3.4 cm"),
    "556": dict(item="AmmoBox556", rows=["556RoundBox"], W=58, D=119, H=74, nx=5, ny=10, count=50,
                round=("round_556", "Round556_Low", "FMJ"), cal="5.56x45 mm", cal_big="5.56", sub="55 GR FMJ",
                kind="CENTERFIRE RIFLE CARTRIDGES", color="#2e6b34", accent="#f2efe6", ink="#ffffff", tray=(0.05, 0.05, 0.05),
                real="50-round 5.56 tuck box with tray, sized to the DT_Ammo row 11.9 x 5.8 x 7.4 cm"),
    "762": dict(item="AmmoBox762", rows=["762RoundBox (new)"], W=58, D=70, H=77, nx=4, ny=5, count=20,
                round=("round_762", "Round762_Low", "FMJ"), cal="7.62x51 mm", cal_big="7.62", sub="147 GR FMJ",
                kind="CENTERFIRE RIFLE CARTRIDGES", color="#6e5a2c", accent="#f2c230", ink="#ffffff", tray=(0.05, 0.05, 0.05),
                real="20-round .308 / 7.62x51 box: 5.8 x 7.0 x 7.7 cm"),
    "792": dict(item="AmmoBox792", rows=["792RoundBox (new)"], W=58, D=70, H=86, nx=4, ny=5, count=20,
                round=("round_792", "Round792_Low", "FMJ"), cal="7.92x57 mm", cal_big="7.92", sub="198 GR FMJ",
                kind="CENTERFIRE RIFLE CARTRIDGES", color="#4f564c", accent="#b3261e", ink="#f2efe6", tray=(0.62, 0.55, 0.42),
                real="20-round 8 mm Mauser box with board divider: 5.8 x 7.0 x 8.6 cm"),
    "12ga": dict(item="AmmoBox12ga", rows=["ShotgunShellBox"], W=106, D=118, H=113, nx=4, ny=5, count=20,
                 round=("shell_12ga", "Shell12ga_Low", "Buck"), cal="12 GAUGE", cal_big="12 GA", sub="00 BUCK · 9 PELLETS",
                 kind="2 3/4 IN SHOTSHELLS", color="#b3261e", accent="#141414", ink="#ffffff", tray=(0.62, 0.55, 0.42),
                 real="20-shell 12 ga box on a board riser, sized to the DT_Ammo row 10.6 x 11.3 x 11.8 cm"),
}


# ---------------------------------------------------------------- geometry
def body(b):
    W, D, H = b["W"], b["D"], b["H"] - TL
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=(W * MM, D * MM, H * MM), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0, 0, H / 2 * MM), verts=bm.verts)
    top = [f for f in bm.faces if f.normal.z > 0.9]
    bmesh.ops.delete(bm, geom=top, context="FACES")
    edges = [e for e in bm.edges if len(e.link_faces) == 2]
    bmesh.ops.bevel(bm, geom=edges, offset=0.5 * MM, segments=2, affect="EDGES", profile=0.5)
    me = bpy.data.meshes.new("SM_" + b["item"])
    bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new(me.name, me)
    bpy.context.scene.collection.objects.link(o)
    s = o.modifiers.new("S", "SOLIDIFY"); s.thickness = T * MM; s.offset = -1.0; s.use_even_offset = True
    geo.apply_mods(o)
    return o


def lid(b):
    """Top panel + tuck flap (inside the -X wall, 0.25 mm clear of it), joined as one mesh with the flap fused in."""
    W, D, H = b["W"], b["D"], b["H"]
    fh = min(16.0, H * 0.45)
    p = geo.rounded_box("lidp", W, D, TL, 0.0, H - TL, 0.0)
    fx = -W / 2 + T + 0.25 + TL / 2
    f = geo.rounded_box("flap", TL, D - 2 * T - 1.2, fh + 0.3, 0.0, H - TL - fh, 0.0, center=(fx, 0))
    geo.boolean(p, f, "UNION")
    p.name = p.data.name = "SM_" + b["item"] + "_Lid"
    bm = bmesh.new(); bm.from_mesh(p.data)
    bmesh.ops.triangulate(bm, faces=[q for q in bm.faces if len(q.verts) > 4], quad_method="BEAUTY")
    bm.to_mesh(p.data); bm.free()
    return p, fh


def tray(b, z_r0):
    """Tray block from the floor to its top (rounds stand in it, bases at z_r0). Clears the walls by 0.4 mm."""
    W, D = b["W"] - 2 * T - 0.8, b["D"] - 2 * T - 0.8
    rl = round_len(b)
    top = z_r0 + 0.45 * rl
    o = geo.rounded_box("SM_" + b["item"] + "_Tray", W, D, top - T, 1.0, T, 0.4, segs=2)
    return o, top, W, D


def round_len(b):                                   # (length, diameter) mm of the round that goes in the box
    rd = json.load(open(os.path.join(AMMO, b["round"][0], "manifest.json")))
    m = rd["meshes"]["SM_" + b["round"][1][:-4]]
    return m["dims_cm"][2] * 10


def round_dia(b):
    rd = json.load(open(os.path.join(AMMO, b["round"][0], "manifest.json")))
    return rd["meshes"]["SM_" + b["round"][1][:-4]]["dims_cm"][0] * 10


def slots(b, tw, td):
    px, py = tw / b["nx"], td / b["ny"]
    return [((i + 0.5) * px - tw / 2, (j + 0.5) * py - td / 2) for j in range(b["ny"]) for i in range(b["nx"])]


# ---------------------------------------------------------------- UVs (dieline): axis-aligned faces → printed rects
def axis_uv(o, mapping, misc_rect, tol=0.3):
    """mapping: key ("+X".. "-Z") → list of (rect, plane_mm, (lo_a, hi_a), (lo_b, hi_b)) in mm. A face whose dominant normal
    is key, whose centre lies on the plane (± tol) and inside the ranges is projected into rect (u follows the viewer's
    right seen from outside); every other face keeps a smart-projected layout squeezed into misc_rect."""
    geo.unwrap_into(o, misc_rect)
    bm = bmesh.new(); bm.from_mesh(o.data)
    uv = bm.loops.layers.uv.active
    for f in bm.faces:
        n = f.normal
        ax = max(range(3), key=lambda i: abs(n[i]))
        if abs(n[ax]) < 0.97:
            continue
        key = ("+" if n[ax] > 0 else "-") + "XYZ"[ax]
        c = f.calc_center_median() / MM
        a, bb = [i for i in range(3) if i != ax]
        for rect, plane, (la, ha), (lb, hb) in mapping.get(key, []):
            if abs(c[ax] - plane) > tol or not (la - 0.01 <= c[a] <= ha + 0.01 and lb - 0.01 <= c[bb] <= hb + 0.01):
                continue
            u0, v0, u1, v1 = rect
            for l in f.loops:
                p = l.vert.co / MM
                ta, tb = (p[a] - la) / (ha - la), (p[bb] - lb) / (hb - lb)
                if key in ("+Y", "-X", "-Z"):
                    ta = 1 - ta
                l[uv].uv = (u0 + (u1 - u0) * min(max(ta, 0), 1), v0 + (v1 - v0) * min(max(tb, 0), 1))
            break
    bm.to_mesh(o.data); bm.free()


def layout(b, fh, tw, td):
    """Shelf-pack every printed panel (mm sizes) at one texel density. Returns {name: rect}, uv-per-mm."""
    W, D, Hb = b["W"], b["D"], b["H"] - TL
    fd = D - 2 * T - 1.2
    items = [("front", D, Hb), ("back", D, Hb), ("left", W, Hb), ("right", W, Hb), ("top", W, D), ("under", W, D),
             ("bottom", W, D), ("flap_out", fd, fh), ("tray_top", tw, td), ("body_misc", D, W + Hb),
             ("lid_misc", D * 0.5, W * 0.5), ("tray_misc", D * 0.6, W * 0.6)]
    gap = 0.008

    def place(k):
        out, x, y, rowh = {}, gap, gap, 0.0
        for nm, w, h in sorted(items, key=lambda t: -t[2]):
            if w * k > 1 - 2 * gap:
                return None
            if x + w * k > 1 - gap:
                x, y, rowh = gap, y + rowh + gap, 0.0
            out[nm] = (x, y, x + w * k, y + h * k)
            x += w * k + gap
            rowh = max(rowh, h * k)
        return out if y + rowh <= 1 - gap else None
    lo, hi = 1e-5, 0.1
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if place(mid) else (lo, mid)
    return place(lo), lo


# ---------------------------------------------------------------- print art (Label, cm)
def emblem(L, cx, cy, r, c1, c2):
    """Invented maker mark: a ring with a five-point star and two chevrons. No real brand."""
    L.ellipse(cx, cy, r, r, c1)
    L.ellipse(cx, cy, r * 0.84, r * 0.84, c2)
    pts = [(cx + r * 0.62 * (1 if k % 2 == 0 else 0.4) * math.cos(-math.pi / 2 + k * math.pi / 5),
            cy + r * 0.62 * (1 if k % 2 == 0 else 0.4) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)]
    L.poly(pts, c1)


def art_front(b, w, h, ppc):
    L = Label(w, h, ppc, bg="#f1eee6")
    bw = min(w * 0.36, h * 1.6)
    L.rect(0, 0, bw, h, b["color"])
    L.text(bw / 2, h * 0.47, b["cal_big"], "display", h * 0.42, b["ink"], "mm", max_w=bw * 0.86)
    L.text(bw / 2, h * 0.8, b["cal"], "sans", h * 0.1, b["ink"], "mm", max_w=bw * 0.86)
    x = bw + h * 0.12
    L.text(x, h * 0.16, "PATRIOT ARMS CO.", "sans", h * 0.11, "#1a1a1a", "lm", tracking=0.06, max_w=w - x - h * 0.1)
    L.text(x, h * 0.44, b["sub"], "display", h * 0.26, "#1a1a1a", "lm", max_w=w - x - h * 0.1)
    L.text(x, h * 0.7, f"{b['count']} {b['kind']}", "sans", h * 0.095, "#1a1a1a", "lm", max_w=w - x - h * 0.1)
    L.rect(bw, h * 0.86, w, h, b["accent"] if b["accent"] != "#f2efe6" else b["color"])
    L.text(w - h * 0.1, h * 0.93, "RANGE · TRAINING · DEFENSE", "sans", h * 0.06, b["ink"] if b["accent"] in ("#141414",) else "#1a1a1a", "rm", max_w=w - bw - 0.2)
    return L


def art_top(b, w, h, ppc):
    """Reader frame: w = box length (D), h = box width (W)."""
    L = Label(w, h, ppc, bg=b["color"])
    L.rect(0, h * 0.78, w, h, "#f1eee6")
    L.rect(0, h * 0.74, w, h * 0.78, b["accent"])
    r = h * 0.2
    emblem(L, r * 1.3, h * 0.32, r, "#f1eee6", b["color"])
    x = r * 2.7
    L.text(x, h * 0.16, "PATRIOT ARMS CO.", "sans", h * 0.075, b["ink"], "lm", tracking=0.12, max_w=w - x - 0.3)
    L.text(x, h * 0.4, b["cal_big"], "display", h * 0.3, b["ink"], "lm", max_w=w - x - 0.3)
    L.text(x, h * 0.61, b["sub"], "sans", h * 0.085, b["ink"], "lm", max_w=w - x - 0.3)
    L.text(w / 2, h * 0.89, f"{b['count']} {b['kind']}", "sans", h * 0.075, "#1a1a1a", "mm", max_w=w * 0.92)
    return L


def art_side(b, w, h, ppc):
    L = Label(w, h, ppc, bg="#f1eee6")
    L.rect(0, 0, w, h * 0.5, b["color"])
    L.text(w / 2, h * 0.27, b["cal_big"], "display", h * 0.3, b["ink"], "mm", max_w=w * 0.88)
    L.text(w / 2, h * 0.64, f"{b['count']} RDS", "display", h * 0.17, "#1a1a1a", "mm", max_w=w * 0.88)
    L.text(w / 2, h * 0.82, b["sub"], "sans", h * 0.07, "#1a1a1a", "mm", max_w=w * 0.88)
    return L


def art_back(b, w, h, ppc, seed):
    L = Label(w, h, ppc, bg="#f1eee6")
    L.rect(0, 0, w, h * 0.12, b["color"])
    fs = max(min(h * 0.055, 0.17), 0.09)
    y = L.para(h * 0.08, h * 0.18, w * 0.62, "WARNING: KEEP OUT OF REACH OF CHILDREN. Use only in firearms in good condition, "
               "chambered for this cartridge. Wear eye and ear protection. Do not expose to open flame. Store cool and dry. "
               "Lead exposure possible: wash hands after handling.", "sans_r", fs, "#1a1a1a", 1.2)
    L.text(h * 0.08, min(y + fs * 0.6, h * 0.9), f"PATRIOT ARMS CO. · LOT PA26-{100 + seed * 7}", "mono_b", fs * 0.95, "#1a1a1a", "lm", max_w=w * 0.62)
    bw = min(w * 0.28, h * 0.9)
    L.barcode(w - bw - h * 0.1, h * 0.25, bw, min(h * 0.45, bw * 0.55), f"08{zlib.crc32(b['item'].encode()) % 10**10:010d}"[:12])
    return L


def art_under(b, w, h, ppc):
    L = Label(w, h, ppc, bg="#f1eee6")
    L.rect(h * 0.06, h * 0.06, w - h * 0.06, h - h * 0.06, None, outline=b["color"], width=h * 0.02)
    L.text(w / 2, h * 0.3, "INSPECT EVERY ROUND BEFORE LOADING", "sans", h * 0.075, "#1a1a1a", "mm", max_w=w * 0.85)
    L.text(w / 2, h * 0.5, f"{b['cal']}  ·  {b['sub']}", "display", h * 0.14, b["color"], "mm", max_w=w * 0.85)
    L.text(w / 2, h * 0.7, "PATRIOT ARMS CO. · SINCE 1961 · BOX MAY BE REUSED", "sans_r", h * 0.06, "#1a1a1a", "mm", max_w=w * 0.85)
    return L


def art_flap(b, w, h, ppc):
    L = Label(w, h, ppc, bg=b["color"])
    L.text(w / 2, h * 0.5, "TUCK IN · PULL TO OPEN", "sans", h * 0.28, b["ink"], "mm", tracking=0.08, max_w=w * 0.9)
    return L


def art_bottom(b, w, h, ppc, seed):
    L = Label(w, h, ppc, bg="#f1eee6")
    L.text(w / 2, h * 0.3, f"{b['cal']} · {b['count']} RDS", "mono_b", h * 0.08, "#1a1a1a", "mm", max_w=w * 0.9)
    L.text(w / 2, h * 0.45, f"LOT PA26-{100 + seed * 7}  ·  PACKED 03/26", "mono", h * 0.06, "#1a1a1a", "mm", max_w=w * 0.9)
    return L


def art_tray(b, tw, td, ppc, sl, hole_r):
    """World frame (u = +X, v = +Y): dark round holes at the slots."""
    L = Label(tw / 10, td / 10, ppc)
    for x, y in sl:
        cx, cy = (x + tw / 2) / 10, (td / 2 - y) / 10
        L.ellipse(cx, cy, hole_r / 10 + 0.04, hole_r / 10 + 0.04, (0.12, 0.11, 0.10, 0.55))
        L.ellipse(cx, cy, hole_r / 10, hole_r / 10, "#0b0b0b")
    return L


def paste(full, art, rect):
    S = full.shape[0]
    u0, v0, u1, v1 = rect
    i0, i1, j0, j1 = int(round((1 - v1) * S)), int(round((1 - v0) * S)), int(round(u0 * S)), int(round(u1 * S))
    im = Image.fromarray((np.clip(art, 0, 1) * 255).astype(np.uint8)).resize((max(j1 - j0, 1), max(i1 - i0, 1)), Image.LANCZOS)
    full[i0:i1, j0:j1] = np.asarray(im, np.float32) / 255


# ---------------------------------------------------------------- build
def build(cal, here):
    b = BOXES[cal]
    item = b["item"]
    texd, fbxd, rend = (os.path.join(here, d) for d in ("textures", "fbx", "renders"))
    work = os.path.join(ROOT, "work", "ammo_box_" + cal)
    os.makedirs(work, exist_ok=True)
    scene.reset()
    col, colc = scene.collection("Game"), scene.collection("Collision")
    W, D, H = b["W"], b["D"], b["H"]
    Hb = H - TL
    rl = round_len(b)
    z_r0 = H - TL - 1.5 - rl                                   # round bases: tips 1.5 mm under the lid
    bo = scene.link(body(b), col)
    lo_, fh = lid(b)
    lo_ = scene.link(lo_, col)
    tr, ttop, tw, td = tray(b, z_r0)
    tr = scene.link(tr, col)
    rects, k = layout(b, fh, tw, td)
    S = 1024 if (W * D + 2 * Hb * (W + D)) < 30000 else 2048
    fd = D - 2 * T - 1.2
    fx = -W / 2 + T + 0.25
    axis_uv(bo, {"-X": [(rects["front"], -W / 2, (-D / 2, D / 2), (0, Hb))], "+X": [(rects["back"], W / 2, (-D / 2, D / 2), (0, Hb))],
                 "-Y": [(rects["left"], -D / 2, (-W / 2, W / 2), (0, Hb))], "+Y": [(rects["right"], D / 2, (-W / 2, W / 2), (0, Hb))],
                 "-Z": [(rects["bottom"], 0.0, (-W / 2, W / 2), (-D / 2, D / 2))]}, rects["body_misc"])
    axis_uv(lo_, {"+Z": [(rects["top"], H, (-W / 2, W / 2), (-D / 2, D / 2))],
                  "-Z": [(rects["under"], H - TL, (-W / 2, W / 2), (-D / 2, D / 2))],
                  "-X": [(rects["flap_out"], fx, (-fd / 2, fd / 2), (H - TL - fh, H - TL))]}, rects["lid_misc"])
    axis_uv(tr, {"+Z": [(rects["tray_top"], ttop, (-tw / 2, tw / 2), (-td / 2, td / 2))]}, rects["tray_misc"])
    for o in (bo, lo_, tr):
        geo.weighted_normals(o)
    # pivots
    geo.set_origin(lo_, (W / 2 * MM, 0, H * MM))
    geo.set_origin(tr, (0, 0, T * MM))
    # ---- bakes: AO per part with the lid open (inside stays readable), the tray in place
    lo_.rotation_euler.y = math.radians(115)
    smp = 8 if FAST else 32
    aos = {}
    for o in (bo, lo_, tr):
        aos[o.name] = bake.bake(o, [], "AO", S, f"{work}/ao_{o.name}.png", samples=smp)
    lo_.rotation_euler.y = 0
    u, v = tex.uv_grid(S)
    own = {bo.name: ["front", "back", "left", "right", "bottom", "body_misc"], lo_.name: ["top", "under", "flap_out", "lid_misc"],
           tr.name: ["tray_top", "tray_misc"]}
    ao = np.ones((S, S), np.float32)
    for nm, keys in own.items():
        m = np.zeros((S, S), bool)
        for kk in keys:
            m |= tex.in_rect(u, v, rects[kk])
        ao = np.where(m, aos[nm], ao)
    ao = np.where(ao < 0.02, 1.0, ao)
    # ---- substrate / masks
    printed = ["front", "back", "left", "right", "bottom", "top", "under", "flap_out"]
    pm = np.zeros((S, S), bool)
    edge = np.zeros((S, S), np.float32)
    for kk in printed:
        r = rects[kk]
        m = tex.in_rect(u, v, r)
        pm |= m
        dmm = np.minimum.reduce([u - r[0], r[2] - u, v - r[1], r[3] - v]) / k        # mm to the panel edge (fold)
        edge = np.where(m, np.clip(1 - dmm / 2.5, 0, 1), edge)
    trm = tex.in_rect(u, v, rects["tray_top"]) | tex.in_rect(u, v, rects["tray_misc"])
    n1, n2 = tex.fbm(S, S, 12, 1), tex.fbm(S, S, 40, 2)
    kraft = np.array((0.42, 0.32, 0.20), np.float32)
    board = np.array((0.78, 0.76, 0.71), np.float32)
    base = np.where(pm[..., None], board, kraft) * (0.92 + 0.12 * n1[..., None])
    rough = np.where(trm, 0.38, 0.78).astype(np.float32)
    metal = np.zeros((S, S), np.float32)
    wear = np.clip(edge * (0.5 + 0.9 * n2) - 0.35, 0, 1) * 1.6 + np.clip(n1 - 0.7, 0, 1) * 1.2 * pm
    hgt = (tex.fbm(S, S, 200, 5) - 0.5) * 0.01 + np.where(pm, -edge ** 3 * 0.15, 0)
    one = np.ones((S, S), np.float32)
    p = lambda kk: os.path.join(texd, f"T_{item}_{kk}.png")
    os.makedirs(texd, exist_ok=True)
    tex.save(tex.lin_to_srgb(base), p("BaseColor"))
    tex.save(tex.height_to_normal(hgt, S * k), p("Normal"))
    tex.save(np.stack([ao, rough, metal], -1), p("ORM"))
    tex.save(np.stack([trm.astype(np.float32), 0 * one, (pm | tex.in_rect(u, v, rects["tray_top"])).astype(np.float32),
                       np.clip(wear, 0, 1)], -1), p("Mask"), "RGBA")
    texs = {kk: p(kk) for kk in ("BaseColor", "Normal", "ORM", "Mask")}
    # ---- print
    ppc = max(60, min(160, int(S * k * 10 * 1.3)))
    full = np.zeros((S, S, 4), np.float32)
    seed = sum(map(ord, cal))
    rot = lambda L: np.rot90(L.array(), -1)
    paste(full, art_front(b, D / 10, Hb / 10, ppc).array(), rects["front"])
    paste(full, art_back(b, D / 10, Hb / 10, ppc, seed).array(), rects["back"])
    paste(full, art_side(b, W / 10, Hb / 10, ppc).array(), rects["left"])
    paste(full, art_side(b, W / 10, Hb / 10, ppc).array(), rects["right"])
    paste(full, rot(art_top(b, D / 10, W / 10, ppc)), rects["top"])
    paste(full, rot(art_under(b, D / 10, W / 10, ppc)), rects["under"])
    paste(full, rot(art_bottom(b, D / 10, W / 10, ppc, seed)), rects["bottom"])
    paste(full, art_flap(b, fd / 10, fh / 10, ppc).array(), rects["flap_out"])
    sl = slots(b, tw, td)
    paste(full, art_tray(b, tw, td, ppc, sl, round_dia(b) / 2 + 0.15).array(), rects["tray_top"])
    full = age(full, seed, fade=0.08, scuff=0.12)
    lp = tex.save(full, os.path.join(texd, f"T_{item}_Label_Default.png"), "RGBA")
    prm = dict(ColorA=b["tray"], RoughA=0.4, MetalA=0.0, LabelRough=0.6, WearAmount=0.45, Dirt=0.45)
    mi = master.material(f"MI_{item}_Default", texs, prm, label=lp)
    mi.use_fake_user = True
    for o in (bo, lo_, tr):
        if o.data.materials:
            o.data.materials[0] = mi
        else:
            o.data.materials.append(mi)
    # ---- collision, export
    h = scene.link(export.hull(bo, f"UCX_SM_{item}_00", 16), colc); h.hide_render = True
    tris = {o.name: scene.tris(o) for o in (bo, lo_, tr)}
    export.fbx([bo, h], f"{fbxd}/SM_{item}.fbx")
    export.fbx([lo_], f"{fbxd}/SM_{item}_Lid.fbx")
    export.fbx([tr], f"{fbxd}/SM_{item}_Tray.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(here, os.path.basename(here) + ".blend"), relative_remap=True)
    hero(b, bo, lo_, tr, sl, z_r0, rend, os.path.basename(here))
    rd = round(D / 10, 2)
    export.manifest(os.path.join(here, "manifest.json"), {
        "item": item, "replaces_rows": b["rows"], "category": "ammo", "clip": None, "real_object": b["real"],
        "dims_cm": [W / 10, D / 10, H / 10], "pivot": "bottom centre, long axis +Y, front (tuck side) faces -X",
        "parts": {f"SM_{item}_Lid": {"pivot": "hinge line, top of the +X wall", "rest_location_cm": [W / 20, 0, H / 10],
                                     "rest_rotation_deg": [0, 0, 0], "opens": "about local +Y, 0 -> 115 deg"},
                  f"SM_{item}_Tray": {"pivot": "tray bottom centre", "rest_location_cm": [0, 0, T / 10], "rest_rotation_deg": [0, 0, 0],
                                      "opens": f"slides +Z, {round((z_r0 + rl * 0.45) / 10 + 0.5, 1)} cm to clear the walls"}},
        "contents": {"round_mesh": f"SM_{b['round'][1]}", "count": b["count"], "printed_count": b["count"],
                     "slot_base_z_cm": round(z_r0 / 10, 3),
                     "slots_cm": [[round(x / 10, 3), round(y / 10, 3)] for x, y in sl]},
        "tris": tris, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {"Default": os.path.basename(lp)}},
        "variant_params": {"Default": prm}, "mask": "R tray plastic · B print · A fold-edge wear",
        "collision": f"UCX_SM_{item}_00 (box hull)", "lods": "none (< 2k tris)"})
    print("TRIS", tris)


def load_round(b):
    """Import the low round FBX and give it its own master material (textures + params from the round's manifest)."""
    folder, mesh, var = b["round"]
    rdir = os.path.join(AMMO, folder)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=os.path.join(rdir, "fbx", f"SM_{mesh}.fbx"))
    new = [o for o in bpy.data.objects if o not in before]
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
    rnd = [o for o in new if not o.name.startswith("UCX_")][0]
    man = json.load(open(os.path.join(rdir, "manifest.json")))
    vd = man["variants"][var]
    texs = {kk: os.path.join(rdir, "textures", f"T_{mesh}_{kk}.png") for kk in ("BaseColor", "Normal", "ORM", "Mask")}
    lab = os.path.join(rdir, "textures", vd["label_low"]) if vd.get("label_low") else None
    m = master.material(f"MI_{mesh}_{var}_box", texs, vd["params"], label=lab)
    rnd.data.materials.clear()
    rnd.data.materials.append(m)
    rnd.rotation_euler = (0, 0, 0)
    rnd.location = (0, 0, 0)
    rnd.scale = (1, 1, 1)
    return rnd


def place_set(b, bo, lo_, tr, rnd, sl, z_r0, M, lid_deg, tray_up=0.0, with_rounds=True):
    out = []
    for src in (bo, lo_, tr):
        d = src.copy(); d.data = src.data
        d.hide_render = False
        bpy.context.scene.collection.objects.link(d)
        out.append(d)
    out[1].rotation_euler.y = math.radians(lid_deg)
    out[2].location.z += tray_up
    bpy.context.view_layer.update()
    for d in out:
        d.matrix_world = M @ d.matrix_world
    if with_rounds:
        for j, (x, y) in enumerate(sl):
            d = rnd.copy()
            d.hide_render = False
            bpy.context.scene.collection.objects.link(d)
            d.matrix_world = M @ Matrix.Translation((x * MM, y * MM, z_r0 * MM + tray_up)) @ Matrix.Rotation(j * 1.7, 4, "Z")
            out.append(d)
    return out


def hero(b, bo, lo_, tr, sl, z_r0, rend, stem):
    rnd = load_round(b)
    W, D, H = b["W"] * MM, b["D"] * MM, b["H"] * MM
    for o in list(bpy.data.collections["Game"].objects) + list(bpy.data.collections["Collision"].objects):
        o.hide_render = True
    # rest (closed) on the left, open with rounds on the right; front (-X) turned toward the camera
    Rz = Matrix.Rotation(math.radians(90), 4, "Z")               # -X front → -Y (toward the camera)
    gap = max(W, D) * 0.25
    A = Matrix.Translation((-(D / 2 + gap / 2), 0, 0)) @ Rz
    B = Matrix.Translation(((D / 2 + gap / 2), 0, 0)) @ Rz
    objs = place_set(b, bo, lo_, tr, rnd, sl, z_r0, A, 0, with_rounds=False)
    objs += place_set(b, bo, lo_, tr, rnd, sl, z_r0, B, 115)
    scene.studio(objs)
    scene.frame(objs, yaw=-20, pitch=28, lens=70, fill=0.9, name="Hero", aspect=640 / 1024)
    scene.render(os.path.join(rend, f"{stem}.png"), 1024, 16 if FAST else 48, res_y=640)
    for o in objs:
        bpy.data.objects.remove(o)
    for o in [o for o in bpy.data.objects if o.type in ("LIGHT", "CAMERA") or o.name.startswith("Backdrop")]:
        bpy.data.objects.remove(o)
    # parts: lid fully open, tray lifted out with its rounds, top-down 3/4
    objs = place_set(b, bo, lo_, tr, rnd, sl, z_r0, Rz, 115, tray_up=(z_r0 / 1000 + round_len(b) * 0.45 * MM + H * 0.35))
    scene.studio(objs)
    scene.frame(objs, yaw=-30, pitch=30, lens=70, fill=0.85, name="Parts", aspect=1.0)
    scene.render(os.path.join(rend, f"{stem}_parts.png"), 768, 16 if FAST else 48)
