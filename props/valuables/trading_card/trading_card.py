"""Stoopball Cards: SM_TradingCard (63.5 × 88.9 × 0.4 mm, slight bow) in SM_Toploader (76 × 102 × 2 mm clear PVC).
12 players + 1 holo; invented teams, invented players.  .venv/bin/python props/valuables/trading_card/trading_card.py [--fast]"""
import os, sys, math, zlib
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy, bmesh
import numpy as np
from lab import scene, master, export, tex, geo
from lab.labels import Label, age

FAST = "--fast" in sys.argv
S = 2048
MM = 0.001
CW, CH = 63.5, 88.9
TEXD, FBXD, REND = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders"))
UV_FRONT, UV_BACK = (0.0, 0.0, 0.49, 0.686), (0.51, 0.0, 1.0, 0.686)
UV_EDGE = (0.0, 0.70, 1.0, 0.72)
UV_TL = (0.0, 0.74, 1.0, 1.0)


def card():
    """Thin slab with a gentle bow; front/back planar islands, edges in a strip."""
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    nx, nz = 6, 8
    def P(i, j, side):
        x, z = -CW / 2 + CW * i / nx, CH * j / nz
        bow = 0.6 * (1 - (2 * i / nx - 1) ** 2)
        return bm.verts.new((x * MM, (side * 0.2 - bow) * MM, z * MM))
    F = [[P(i, j, -1) for i in range(nx + 1)] for j in range(nz + 1)]
    B = [[P(i, j, 1) for i in range(nx + 1)] for j in range(nz + 1)]
    def quad(vs, rect, ij, flip):
        f = bm.faces.new(vs)
        u0, v0, u1, v1 = rect
        for l, (i, j) in zip(f.loops, ij):
            uu = i / nx
            l[uv].uv = (u0 + (u1 - u0) * ((1 - uu) if flip else uu), v0 + (v1 - v0) * j / nz)
    for j in range(nz):
        for i in range(nx):
            quad((F[j][i], F[j][i + 1], F[j + 1][i + 1], F[j + 1][i]), UV_FRONT, ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)), False)
            quad((B[j][i + 1], B[j][i], B[j + 1][i], B[j + 1][i + 1]), UV_BACK, ((i + 1, j), (i, j), (i, j + 1), (i + 1, j + 1)), True)
    ring = [(i, 0) for i in range(nx)] + [(nx, j) for j in range(nz)] + [(i, nz) for i in range(nx, 0, -1)] + [(0, j) for j in range(nz, 0, -1)]
    for k in range(len(ring)):
        (i0, j0), (i1, j1) = ring[k], ring[(k + 1) % len(ring)]
        f = bm.faces.new((F[j0][i0], B[j0][i0], B[j1][i1], F[j1][i1]))
        u0, v0, u1, v1 = UV_EDGE
        for l, t in zip(f.loops, (k, k, k + 1, k + 1)):
            l[uv].uv = (u0 + (u1 - u0) * t / len(ring), v0 + (v1 - v0) * (0 if l.vert in [F[j0][i0], F[j1][i1]] else 1))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new("SM_TradingCard"); bm.to_mesh(me); bm.free()
    o = bpy.data.objects.new("SM_TradingCard", me); bpy.context.scene.collection.objects.link(o)
    return o


def toploader():
    o = geo.rounded_box("SM_Toploader", 76.0, 2.6, 101.6, 1.0, -6.0, 0.4, segs=2)
    cav = geo.rounded_box("pocket", 66.5, 1.4, 101.0, 0.5, -2.5, 0.0, segs=2)          # open at the top
    geo.boolean(o, cav)
    geo.unwrap_into(o, UV_TL)
    return o


TEAMS = {"Gowanus Gulls": ("#1f5c8f", "#f2c21a"), "Queens Comets": ("#d2482a", "#1d2b4a"), "Harlem Hawks": ("#2a2a2a", "#c9a54a"),
         "Bronx Iron": ("#6a6a70", "#c8202a"), "Brooklyn Bridgers": ("#2a7a4a", "#f2f2ee"), "Staten Ferrymen": ("#e88a1a", "#1f3a6a")}
PLAYERS = [("Sal 'The Spigot' Marino", "Gowanus Gulls", "P", "bat"), ("Dee Okafor", "Queens Comets", "SS", "field"),
           ("Manny Ruiz", "Harlem Hawks", "CF", "bat"), ("Tommy Kowalczyk", "Bronx Iron", "1B", "bat"),
           ("Lou 'Hot Corner' Bianchi", "Brooklyn Bridgers", "3B", "field"), ("Ray Washington", "Staten Ferrymen", "P", "pitch"),
           ("Kenji Arai", "Queens Comets", "2B", "bat"), ("Big Ed Doyle", "Bronx Iron", "C", "field"),
           ("Vinnie Caruso", "Gowanus Gulls", "LF", "bat"), ("Hector 'Hex' Alvarez", "Harlem Hawks", "P", "pitch"),
           ("Abe Silverman", "Brooklyn Bridgers", "RF", "bat"), ("Jo Ellen Pike", "Staten Ferrymen", "SS", "field"),
           ("Sal Marino ROOKIE HOLO", "Gowanus Gulls", "P", "pitch")]


def silhouette(L, x, y, s, pose, col):
    """A simple player figure in three poses (bat / pitch / field)."""
    L.ellipse(x, y - 2.6 * s, 0.45 * s, 0.5 * s, col)
    L.poly([(x - 0.7 * s, y - 2.0 * s), (x + 0.7 * s, y - 2.0 * s), (x + 0.55 * s, y + 0.3 * s), (x - 0.55 * s, y + 0.3 * s)], col)
    leg = 0.9 if pose != "field" else 1.4
    L.poly([(x - 0.5 * s, y + 0.2 * s), (x - 0.1 * s, y + 0.2 * s), (x - leg * s, y + 2.4 * s), (x - (leg + 0.4) * s, y + 2.4 * s)], col)
    L.poly([(x + 0.1 * s, y + 0.2 * s), (x + 0.5 * s, y + 0.2 * s), (x + (leg + 0.4) * s, y + 2.4 * s), (x + leg * s, y + 2.4 * s)], col)
    if pose == "bat":
        L.line([(x + 0.4 * s, y - 1.5 * s), (x - 1.6 * s, y - 3.6 * s)], "#8a5a2a", 0.14 * s)
    elif pose == "pitch":
        L.line([(x + 0.6 * s, y - 1.8 * s), (x + 1.8 * s, y - 3.0 * s)], col, 0.3 * s); L.ellipse(x + 1.9 * s, y - 3.15 * s, 0.18 * s, 0.18 * s, "#ffffff")
    else:
        L.line([(x - 0.6 * s, y - 1.6 * s), (x - 1.7 * s, y - 0.8 * s)], col, 0.3 * s); L.ellipse(x - 1.8 * s, y - 0.7 * s, 0.35 * s, 0.3 * s, "#5a3a1a")


def card_art(i, name, team, pos, pose):
    c1, c2 = TEAMS[team]
    holo = "HOLO" in name
    F = Label(CW / 10, CH / 10, 280, bg=c1)
    F.rect(0.35, 0.35, CW / 10 - 0.35, CH / 10 - 1.8, "#9ec6e8" if not holo else "#c8c8d8")
    F.gradient(0.35, 0.35, CW / 10 - 0.35, 4.5, "#7fb0d8" if not holo else "#ffd0ff", "#cfe4f2" if not holo else "#d0fff0")
    F.rect(0.35, 4.6, CW / 10 - 0.35, CH / 10 - 1.8, "#4a8a3a")                    # outfield grass
    F.ellipse(CW / 20, 6.6, 2.2, 0.6, "#b88a5a")                                    # infield dirt
    silhouette(F, CW / 20, 5.2, 0.85, pose, c2)
    F.rect(0, CH / 10 - 1.85, CW / 10, CH / 10, c2)
    F.text(CW / 20, CH / 10 - 1.25, name.replace(" ROOKIE HOLO", "").upper(), "sans", 0.3, c1, "mm", max_w=5.6)
    F.text(CW / 20, CH / 10 - 0.65, f"{team.upper()}  ·  {pos}", "sans_r", 0.2, c1, "mm", max_w=5.6)
    F.ellipse(5.45, 1.0, 0.55, 0.55, c2); F.text(5.45, 1.0, "87", "heavy", 0.3, c1, "mm")
    if holo:
        F.text(1.1, 1.0, "ROOKIE", "heavy", 0.26, "#ffffff", "mm", rotate=20)
    B = Label(CW / 10, CH / 10, 280, bg="#e9dfc6")
    B.rect(0, 0, CW / 10, 1.3, c1)
    B.text(0.4, 0.65, f"No. {i + 1:02d}", "heavy", 0.36, "#ffffff", "lm")
    B.text(CW / 10 - 0.4, 0.65, "STOOPBALL CARDS '87", "sans", 0.2, "#ffffff", "rm")
    B.text(0.4, 1.8, name.replace(" ROOKIE HOLO", ""), "serif_b", 0.36, "#2a1d12", "lm", max_w=5.6)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    y = 2.6
    hdr = ["YR", "G", "AB", "H", "HR", "RBI", "AVG"]
    for k, h in enumerate(hdr):
        B.text(0.4 + k * 0.8, y, h, "mono_b", 0.2, "#2a1d12", "lm")
    for yr in range(1981, 1987):
        y += 0.42
        ab, hh = int(rng.integers(380, 610)), 0
        hh = int(ab * rng.uniform(0.22, 0.33))
        row = [str(yr)[2:], str(int(rng.integers(120, 162))), str(ab), str(hh), str(int(rng.integers(3, 38))), str(int(rng.integers(30, 110))), f".{int(hh / ab * 1000):03d}"]
        for k, val in enumerate(row):
            B.text(0.4 + k * 0.8, y, val, "mono", 0.2, "#2a1d12", "lm")
    B.para(0.4, y + 0.6, 5.5, "Played stoopball on Union Street until the cops came. Still holds the borough record for broken windows.", "serif_i", 0.2, "#2a1d12", 1.2)
    B.text(CW / 20, CH / 10 - 0.4, "© 1987 Stoopball Card Co., Brooklyn NY · Printed in USA", "sans_r", 0.14, "#2a1d12", "mm")
    full = np.zeros((S, S, 4), np.float32)
    from PIL import Image
    for lab, rect in ((F, UV_FRONT), (B, UV_BACK)):
        art = age(lab.array(), i, fade=0.06, scuff=0.06)
        u0, v0, u1, v1 = rect
        j0, j1, i0, i1 = int(u0 * S), int(u1 * S), int((1 - v1) * S), int((1 - v0) * S)
        full[i0:i1, j0:j1] = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0), Image.LANCZOS), np.float32) / 255
    full[int((1 - UV_EDGE[3]) * S):int((1 - UV_EDGE[1]) * S)] = [0.9, 0.88, 0.82, 1.0]
    return full, holo


def main():
    scene.reset()
    col = scene.collection("Game")
    c, t = scene.link(card(), col), scene.link(toploader(), col)
    u, v = tex.uv_grid(S)
    tl = tex.in_rect(u, v, UV_TL)
    one = np.ones((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_TradingCard_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    scuff = tex.scratches(S, S, 400, (0.004, 0.04), 1, 3)
    tex.save(tex.lin_to_srgb(np.zeros((S, S, 3)) + 0.85), p("BaseColor"))
    tex.save(tex.height_to_normal(np.where(tl, scuff * 0.02, (tex.fbm(S, S, 60, 2) - 0.5) * 0.01), S / 100), p("Normal"))
    tex.save(np.stack([one, np.where(tl, 0.05 + 0.4 * scuff, 0.45), 0 * one], -1), p("ORM"))
    front = tex.in_rect(u, v, UV_FRONT).astype(np.float32)
    tex.save(np.stack([front, 0 * one, (~tl).astype(np.float32), 0 * one], -1), p("Mask"), "RGBA")
    texs = {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
    pvc = master.glass("M_Toploader", (0.97, 0.98, 0.99), 0.05, 1.5, normal=texs["Normal"], orm=texs["ORM"])
    mats, coll = {}, {}
    for i, (name, team, pos, pose) in enumerate(PLAYERS):
        k = f"{i + 1:02d}_" + ("MarinoHolo" if "HOLO" in name else name.split()[-1])
        art, holo = card_art(i, name, team, pos, pose)
        lp = tex.save(art, os.path.join(TEXD, f"T_TradingCard_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=(0.85, 0.75, 0.95) if holo else (0.5, 0.5, 0.5), RoughA=0.15 if holo else 0.4, MetalA=0.9 if holo else 0.0,
                   LabelRough=0.25 if holo else 0.45, WearAmount=0.0, Dirt=0.15)
        if holo:                                                       # holo: the print goes half-transparent so foil shines through
            art2 = art.copy(); art2[..., 3] *= np.where(tex.in_rect(u, v, UV_FRONT), 0.7, 1.0)
            lp = tex.save(art2, lp, "RGBA")
        m = master.material(f"MI_TradingCard_{k}", texs, prm, label=lp)
        m.use_fake_user = True
        mats[k] = (m, prm, lp)
        coll[k] = dict(number=i + 1, player=name, team=team, rarity="legendary" if holo else ("rare" if i % 5 == 0 else "common"))
    first = next(iter(mats))
    c.data.materials.append(mats[first][0]); t.data.materials.append(pvc)
    tri = {o.name: scene.tris(o) for o in (c, t)}
    export.fbx([c], f"{FBXD}/SM_TradingCard.fbx")
    h = scene.link(export.hull(t, "UCX_SM_Toploader_00", 16), scene.collection("Collision")); h.hide_render = True
    export.fbx([t, h], f"{FBXD}/SM_Toploader.fbx")
    t.location.x = 0.1
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "trading_card.blend"), relative_remap=True)
    t.location.x = 0
    s = bpy.context.scene
    s.cycles.transmission_bounces = s.cycles.max_bounces = 8
    dups = []
    for i, k in enumerate(list(mats)[:6] + [list(mats)[-1]]):
        for o in (c, t) if i == 0 else (c,):
            d = o.copy(); d.data = o.data.copy()
            if o is c:
                d.data.materials[0] = mats[k][0]
            d.rotation_euler = (math.radians(-90), 0, 0)
            d.location = ((i - 3) * 0.072, 0, 0.002)
            bpy.context.scene.collection.objects.link(d)
            dups.append(d)
    c.hide_render = t.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=0, pitch=70, lens=70, fill=0.9, name="Cards", aspect=700 / 1600)
    scene.render(f"{REND}/trading_cards.png", 1600, 16 if FAST else 96, res_y=700)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "TradingCard", "set": "Stoopball Cards", "category": "valuables", "clip": None,
        "dims_cm": {"SM_TradingCard": [6.35, 0.1, 8.89], "SM_Toploader": [7.6, 0.26, 10.16]}, "pivot": "bottom centre (card), slides out of the toploader along +Z",
        "tris": tri, "textures": {"size": S, "normal": "OpenGL (flip green in UE)", "variants": {k: os.path.basename(v[2]) for k, v in mats.items()}},
        "variant_params": {k: v[1] for k, v in mats.items()}, "collection": coll})
    print("TRIS", tri)


if __name__ == "__main__":
    main()
