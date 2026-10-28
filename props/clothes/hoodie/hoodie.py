"""SK_Hoodie: pullover hoodie (Shirt slot) for LootNShoot, rigged to the UE5 Mannequin core + twist bones.
    .venv/bin/python props/clothes/hoodie/hoodie.py [samples=48] [rebake=0] [renders=0]

Builds from nothing: proxy body + tailoring fields (hoodie_body), garment geometry at two resolutions (hoodie_geo: level 2 game
mesh, level 4 bake source with the full fold set), UVs (hoodie_uv), high->low normal / AO bakes, texel-space fabric detail
and masks (hoodie_tex), heat weights on the proxy transferred + cleaned (hoodie_rig), LODs, FBX + re-import check, QC
deformation renders, hero, manifest. Scene units: 1 BU = 1 cm (scale_length 0.01), the pill-bottle clip's convention."""
import os, sys, time, json, math, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import bpy, bmesh
import numpy as np
from mathutils import Vector
from lab import scene, master, bake, tex
import hoodie_body as B, hoodie_geo as G, hoodie_uv as U, hoodie_tex as TX, hoodie_rig as RG

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SMP = min(int(ARGS.get("samples", 48)), 64)
REBAKE = ARGS.get("rebake", "1") != "0"
RENDERS = ARGS.get("renders", "1") != "0"
S = 2048
TEXD, FBXD, REND = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders"))
WORK = os.path.join(ROOT, "work", "hoodie")
ITEM = "Hoodie"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


VARIANTS = {   # sRGB hex: fabric (Mask R), lining + drawstrings (Mask G)
    "HeatherGrey": ("#6E6D69", "#BDB9AF", None),
    "Black": ("#1D1D1F", "#4C4B4A", None),
    "Olive": ("#4E5236", "#B6AC8E", None),
    "Navy": ("#1E2840", "#A3A6AB", None),
    "Maroon": ("#6E1E26", "#7A2630", None),           # matches the user's reference: cords and lining in the body colour
    "NavyPrint": ("#1E2840", "#A3A6AB", "Print"),
}


def lin(hx):
    return tuple(float(x) for x in tex.srgb_to_lin(tex.hex2rgb(hx)))


FIT_HIST = []
FITTED = {}          # variant -> (ColorA, ColorB) linear, measured fit under the studio light (fit_colour)


def vparams(v):
    a, b, _ = VARIANTS[v]
    ca, cb = FITTED.get(v, (lin(a), lin(b)))
    return dict(ColorA=tuple(ca), RoughA=0.93, MetalA=0.0, ColorB=tuple(cb), RoughB=0.86, MetalB=0.0, LabelRough=0.7,
                WearAmount=0.35 if v != "Black" else 0.45, Dirt=0.35)


# ------------------------------------------------------------------------------------------------------------- geometry
def triangulate(o):
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(o.data); bm.free()


def fabric_coords(M, L):
    """Per-loop fabric coordinates (fu across the grain, fv along it, cm) for the texel patterns."""
    V = M["V"]; n_out = M["n_outer"]; nb = M["hood_base"]
    nv = len(V)
    shell_end = 2 * n_out
    base = np.arange(nv) % n_out
    sl = np.zeros(nv); sl[:shell_end] = M["sl"][base[:shell_end]]
    # hood params per shell vertex (row 0 = neckline: angle around, u = 0)
    hphi = np.full(nv, np.nan); hu = np.zeros(nv)
    hood = M["hood"]; idx = hood["idx"]; ncol = idx.shape[0] - 1
    p = hood["params"]
    hv = np.arange(nb, n_out)
    for off in (0, n_out):
        hphi[hv + off] = p["phi"]; hu[hv + off] = p["u"]
        for j in range(ncol):
            hphi[idx[j, 0] + off] = 2 * np.pi * j / ncol; hu[idx[j, 0] + off] = 0
    # torso
    x, y, z = V.T
    cy = np.interp(z, B.BODY_T[:, 0], B.BODY_T[:, 3])
    rr = np.interp(z, B.BODY_T[:, 0], B.BODY_T[:, 1] + B.BODY_T[:, 2]) / 2 + 1.5
    th_f = np.arctan2(x, -(y - cy)); th_b = np.arctan2(x, (y - cy))
    # sleeves
    s_fv = np.zeros(nv); s_fu = np.zeros(nv)
    for s, sg in (("l", 1), ("r", -1)):
        m = np.sign(x) == sg
        sarc, r, l1, l2 = G.arm_coords(V[m], s)
        sh, el, wr = G.arm_axis(s)
        ax = (el - sh) / l1 + (wr - el) / l2; ax /= np.linalg.norm(ax)
        under = np.array([-sg, 0.0, -1.0]); under -= (under @ ax) * ax; under /= np.linalg.norm(under)
        e1 = -under; e2 = np.cross(ax, e1)
        s_fv[m] = sarc; s_fu[m] = np.arctan2(r @ e2, r @ e1) * 6.3
    tuv = M["tube_uv"]
    reg = L["reg"]; ls, lt, lv = L["ls"], L["lt"], L["lv"]
    out = np.zeros((len(lv), 2))
    pk0, pk1 = M["ranges"][1][:2]
    for fi in range(len(ls)):
        lo = np.arange(ls[fi], ls[fi] + lt[fi]); vv = lv[lo]; r_ = reg[fi]
        if r_ in (G.R_STRING, G.R_KNOT, G.R_AGLET):
            c = tuv[vv].copy()
            circ = 2 * np.pi * 0.26
            c[:, 0] = np.where(c[:, 0].max() - c[:, 0] > circ / 2, c[:, 0] + circ, c[:, 0])
            out[lo] = c
        elif r_ == G.R_GROM:
            out[lo] = 0
        elif pk0 <= vv[0] < pk1:
            out[lo] = np.stack([x[vv], z[vv]], 1)
        elif r_ in (G.R_HOOD, G.R_CHAN, G.R_LINING) or (r_ == G.R_RIM and np.all(~np.isnan(hphi[vv])) and np.all(base[vv] >= nb)):
            ph = np.nan_to_num(hphi[vv])
            ph = np.where(ph.max() - ph > np.pi, ph + 2 * np.pi, ph)
            out[lo] = np.stack([ph * 9.0, hu[vv] * 30.0], 1)
        elif r_ in (G.R_SLEEVE, G.R_CUFF) or sl[vv].mean() > 0.5:
            out[lo] = np.stack([s_fu[vv], s_fv[vv]], 1)
        else:
            front = np.mean(np.cos(th_f[vv])) > 0
            th = th_f[vv] if front else th_b[vv]
            out[lo] = np.stack([th * rr[vv], z[vv]], 1)
    return out


def build_meshes():
    log("geometry: game mesh (level 2)")
    lowM = G.build(2, log=log)
    log("geometry: bake source (level 4)")
    hiM = G.build(4, eyelets=lowM["eyelets"], paths=lowM["paths"], log=log)
    game = scene.collection("Game")
    low = U.make_mesh("SK_Hoodie", lowM, game)
    log("uv layout")
    lay = U.layout(low, lowM)
    triangulate(low)
    L = U.read_layout(low)
    mat = bpy.data.materials.new("M_Hoodie")
    low.data.materials.append(mat)
    work = scene.collection("Work")
    high = []
    for g, (a, b) in enumerate(groups(hiM)):
        F = [tuple(i - a for i in f) for f in hiM["F"] if a <= f[0] < b]
        R = [r for f, r in zip(hiM["F"], hiM["reg"]) if a <= f[0] < b]
        o = U.make_mesh(f"Hoodie_High_{GROUPS[g]}", dict(V=hiM["V"][a:b], F=F, reg=R, seams=set()), work, uv=False)
        o.data.materials.append(bpy.data.materials.new(f"M_High_{g}"))
        high.append(o)
    log(f"low {len(low.data.polygons)} tris, high {sum(len(o.data.polygons) for o in high)} polys, {lay['uv_per_cm'] * S:.1f} px/cm")
    return lowM, hiM, low, high, L, lay


GROUPS = ("shell", "pocket", "trims")


def groups(M):
    """Vertex ranges of the bake groups: cloth shell, pocket panel, eyelets + cords (baked separately so a ray that
    starts under a cord or the pocket never hits the wrong part)."""
    r = M["ranges"]
    return [(r[0][0], r[0][1]), (r[1][0], r[1][1]), (r[2][0], r[-1][1])]


# ------------------------------------------------------------------------------------------------------------- textures
def bakes(low, high, lowM, L):
    """Normal + AO per bake group, composited by which group owns each texel's face."""
    os.makedirs(WORK, exist_ok=True)
    bpy.context.scene.world = bpy.context.scene.world or bpy.data.worlds.new("W")
    bpy.context.scene.world.light_settings.distance = 6.0
    T, fid = U.tris_of(L, None, None)
    _, fimg = U.raster(L, np.zeros((len(L["lv"]), 1)), S)
    rng = groups(lowM)
    face_v = L["lv"][L["ls"]]
    face_g = np.zeros(len(L["ls"]), int)
    for g, (a, b) in enumerate(rng):
        face_g[(face_v >= a) & (face_v < b)] = g
    gimg = np.where(fimg >= 0, face_g[np.maximum(fimg, 0)], -1)
    # padding: empty texels take the nearest group
    gimg = U.dilate(gimg, fimg >= 0)
    nrm = np.zeros((S, S, 3), np.float32); ao = np.zeros((S, S), np.float32)
    # several cage extrusions per group; per texel the first that hit a front face wins (layers closer than one cage, fold
    # peaks taller than another), AO from the middle cage
    EXT = {0: (0.8, 0.4, 1.3, 0.2), 1: (0.3, 0.6, 0.15), 2: (0.3, 0.12, 0.6)}
    stats = {}
    for g, hg in enumerate(high):
        m = gimg == g
        pick = np.zeros((S, S), bool)
        for k, ex in enumerate(EXT[g]):
            pn = os.path.join(WORK, f"bake_normal_{GROUPS[g]}_{k}.png")
            if REBAKE or not os.path.exists(pn):
                log(f"bake {GROUPS[g]}: normal, cage {ex} cm")
                n_ = bake.bake(low, [hg], "NORMAL", S, pn, samples=4, extrusion=ex, max_ray=2 * ex + 0.4)
            else:
                n_ = tex.load(pn)[..., :3]
            good = m & ~pick & (n_[..., 2] > 0.55)
            nrm[good] = n_[good]; pick |= good
        rest = m & ~pick
        nrm[rest] = (0.5, 0.5, 1.0)
        stats[GROUPS[g]] = int(rest.sum())
        pa = os.path.join(WORK, f"bake_ao_{GROUPS[g]}.png")
        if REBAKE or not os.path.exists(pa):
            log(f"bake {GROUPS[g]}: AO")
            a_ = bake.bake(low, [hg], "AO", S, pa, samples=24, extrusion=EXT[g][1], max_ray=2 * EXT[g][1] + 0.4)
        else:
            a_ = tex.load(pa); a_ = a_[..., 0] if a_.ndim == 3 else a_
        ao[m] = a_[m]
    log(f"texels with no valid hit at any cage (flat normal): {stats}")
    log("bakes composited")
    return nrm, ao


def textures(M, L, lay, nrm, ao):
    log("raster: positions, fabric coords")
    V = M["V"]
    fc = fabric_coords(M, L)
    lattr = np.concatenate([V[L["lv"]], fc], 1)
    img, fimg = U.raster(L, lattr, S)
    valid = fimg >= 0
    reg = np.where(valid, L["reg"][np.maximum(fimg, 0)], -1)
    R = dict(reg=reg, pos=img[..., :3], fu=img[..., 3], fv=img[..., 4], valid=valid)
    px_cm = lay["uv_per_cm"] * S
    # seams (outer surface), wear sources
    Vo = M["outer"][0]
    pV = M["V"]
    seams = dict(typed=M["sew"], pk_typed={k: v for k, v in M["pk_sew"].items()}, pV=pV)
    body, hood, pk = M["body"], M["hood"], M["pocket"]
    idx = hood["idx"]; rows = idx.shape[1]
    hood_edge = np.concatenate([idx[:, rows - 1], idx[0, :], idx[-1, :]])
    open_v = [a for (a, b), t in M["pk_sew"].items() if t == "opening"] + [b for (a, b), t in M["pk_sew"].items() if t == "opening"]
    elbows = []
    for s, sg in (("l", 1), ("r", -1)):
        m = (np.sign(Vo[:, 0]) == sg) & (np.r_[M["sl"]] > 0.5)
        sarc, r, l1, l2 = G.arm_coords(Vo[m], s)
        sh, el, wr = G.arm_axis(s)
        ud = (el - sh) / l1; fd = (wr - el) / l2
        inner = fd - (fd @ ud) * ud; inner /= np.linalg.norm(inner)
        sel = (np.abs(sarc - l1) < 4.5) & (r @ inner < -0.2)
        elbows.append(Vo[m][sel])
    wear_pts = [(Vo[body["loops"][1][0]], 1.3, 0.75), (Vo[body["loops"][2][0]], 1.2, 0.75), (Vo[body["loops"][3][0]], 1.2, 0.75),
                (Vo[hood_edge], 1.0, 0.6), (pV[np.array(open_v)], 0.9, 0.65), (np.concatenate(elbows), 2.6, 0.45)]
    log("texel detail: knit, ribbing, seams, stitches, wear")
    H, stitches, wear, fuzz = TX.details(R, px_cm, seams, V, wear_pts)
    # per-texel px/cm (inside fabric is laid out at lower density)
    dens = np.ones((S, S), np.float32) * px_cm
    for r_, k in U.DENSITY.items():
        dens[reg == r_] = px_cm * k
    H = U.dilate(H, valid)
    dens = U.dilate(dens, valid)
    detail_n = tex.height_to_normal(H * dens, 1.0)
    nrm_f = tex.blend_normals(nrm, detail_n)
    nrm_f = np.where(valid[..., None], nrm_f, nrm)
    # curvature from the baked normal (fold ridges brighten / rub first)
    nx, ny = nrm[..., 0] * 2 - 1, nrm[..., 1] * 2 - 1
    curv = (np.roll(nx, -1, 1) - np.roll(nx, 1, 1)) - (np.roll(ny, -1, 0) - np.roll(ny, 1, 0))
    curv = tex.blur(curv, 2.0)
    ridge = tex.remap(-curv, 0.02, 0.25)
    # masks
    fab = np.isin(reg, TX.FABRIC + TX.RIB + [G.R_RIM])
    ins = np.isin(reg, TX.INSIDE)
    lining = reg == G.R_LINING
    cord = np.isin(reg, TX.CORD)
    metal = np.isin(reg, TX.METAL)
    heather = tex.fbm(S, S, 140, 21) * 0.6 + tex.fbm(S, S, 600, 22) * 0.4
    mottle = tex.fbm(S, S, 60, 31)
    # fleece sheen, baked: M_Lab_Prop has no sheen input, so fold ridges (where grazing light catches the brushed fibres)
    # mix 6 % towards the lighter substrate; the Blender preview adds Principled sheen on top
    mR = fab * (0.965 + 0.035 * heather) * (1 - 0.05 * ridge) * (1 - 0.28 * np.clip(stitches, 0, 1)) + ins * (0.64 + 0.12 * mottle)
    mG = lining * (0.84 + 0.16 * mottle) + cord * 1.0
    label, area = TX.chest_print(R["pos"], reg, valid, S, os.path.join(ROOT, "fonts"))
    mB = area
    wearA = np.clip(wear + 0.2 * ridge * (fab | lining), 0, 1) * (fab | lining | cord | ins)
    # substrate (what wear fades towards): bleached cotton; gunmetal eyelets; nickel aglets
    sub = np.zeros((S, S, 3), np.float32)
    faded = np.array(tex.srgb_to_lin(tex.hex2rgb("#B9B6AE")))
    sub[:] = faded * (0.88 + 0.12 * fuzz[..., None])
    sub[reg == G.R_GROM] = tex.srgb_to_lin(tex.hex2rgb("#4A4744"))
    sub[reg == G.R_AGLET] = tex.srgb_to_lin(tex.hex2rgb("#9C9A96"))
    grime = tex.remap(1 - ao, 0.1, 0.6) * 0.25
    sub *= (1 - grime[..., None])
    bc = tex.lin_to_srgb(U.dilate(sub, valid))
    # ORM
    aoh = np.clip(ao * (1 - 0.6 * tex.remap(-H, 0.02, 0.08)), 0, 1)
    rough = np.full((S, S), 0.9, np.float32) - 0.05 * fuzz + 0.04 * stitches
    rough[cord] = 0.82
    rough[reg == G.R_GROM] = 0.38; rough[reg == G.R_AGLET] = 0.3
    metal_c = metal.astype(np.float32)
    orm = np.stack([U.dilate(aoh, valid), U.dilate(rough, valid), U.dilate(metal_c, valid)], -1)
    mask = np.stack([U.dilate(mR, valid), U.dilate(mG, valid), U.dilate(mB, valid), U.dilate(wearA, valid)], -1)
    label = U.dilate(label, valid)
    os.makedirs(TEXD, exist_ok=True)
    out = {}
    out["BaseColor"] = tex.save(bc, f"{TEXD}/T_{ITEM}_BaseColor.png")
    out["Normal"] = tex.save(nrm_f, f"{TEXD}/T_{ITEM}_Normal.png")
    out["ORM"] = tex.save(orm, f"{TEXD}/T_{ITEM}_ORM.png")
    out["Mask"] = tex.save(mask, f"{TEXD}/T_{ITEM}_Mask.png", "RGBA")
    lab = tex.save(label, f"{TEXD}/T_{ITEM}_Label_Print.png", "RGBA")
    log("textures written")
    return out, lab, R


def materials(texs, lab, low):
    mats = {}
    for v, (_, _, lb) in VARIANTS.items():
        mats[v] = master.material(f"MI_{ITEM}_{v}", texs, vparams(v), label=lab if lb else None)
        mats[v].use_fake_user = True                     # keep the variant previews in the .blend
    m = master.material(f"M_{ITEM}", texs, vparams("HeatherGrey"))
    low.data.materials[0] = m
    for mm in list(mats.values()) + [m]:
        sheen(mm)
    return mats


def sheen(m):
    """Blender preview only: Principled sheen for the brushed-fleece fuzz (UE: Cloth shading model / fuzz on the MI)."""
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Sheen Weight"].default_value = 0.3
    b.inputs["Sheen Roughness"].default_value = 0.45
    b.inputs["Sheen Tint"].default_value = (1, 1, 1, 1)


# ------------------------------------------------------------------------------------------------------------- skinning
def skin(M, low, rig):
    log("skinning: proxy body + heat weights")
    proxy = B.proxy_mesh("Proxy_Body", res=1.0)
    for _ in range(1):
        sm = proxy.modifiers.new("s", "SMOOTH"); sm.iterations = 4; sm.factor = 0.5
        bpy.context.view_layer.objects.active = proxy; bpy.ops.object.modifier_apply(modifier="s")
    names = RG.ALLOWED
    allnames = [b.name for b in rig.data.bones if b.use_deform]
    pV = np.array([v.co[:] for v in proxy.data.vertices])
    pT = np.array([p.vertices[:] for p in proxy.data.polygons])
    Wall = RG.bone_heat(pV, pT, allnames, B.body_sdf)
    RG.write_groups(proxy, Wall, allnames)
    log(f"  proxy bone-heat weights: {int((Wall.max(0) > 0).sum())} bones")
    Wp = Wall[:, [allnames.index(n) for n in names]]
    stray = Wall[:, [i for i, n in enumerate(allnames) if n not in names]].sum(1)
    Wp /= np.maximum(Wp.sum(1, keepdims=True), 1e-9)
    V = M["V"]; n_out = M["n_outer"]
    Vo, Fo = M["outer"]
    W = np.zeros((len(V), len(names)))
    # garment outer surface: nearest proxy point; cleanup on the outer surface mesh
    Wo = RG.nearest_interp(pV, pT, Wp, Vo)
    E = G.edges_of(np.asarray(Fo))[0]
    Wo = RG.cleanup(Wo, E, smooth=int(ARGS.get("wsmooth", 1)))
    # armholes: extra local smoothing so the armhole edge partly follows the arm (arms forward / up without the swinging
    # arm entering the chest side)
    near = np.zeros(len(Vo), bool)
    for sg in (1, -1):
        near |= np.linalg.norm(Vo - np.array([sg * 20.6, 0.0, 132.0]), axis=1) < 10.0
    deg = np.bincount(E.ravel(), minlength=len(Vo)).astype(float)[:, None]
    for _ in range(int(ARGS.get("apsmooth", 12))):
        acc = np.zeros_like(Wo); np.add.at(acc, E[:, 0], Wo[E[:, 1]]); np.add.at(acc, E[:, 1], Wo[E[:, 0]])
        Wo = np.where(near[:, None], 0.5 * Wo + 0.5 * acc / np.maximum(deg, 1), Wo)
    # elbows: soften the upper/lower blend (w^0.3, measured best vs 3.0 / 0.5) so the bent sleeve stays off the arm
    # into the arm through the linear-blend midpoint
    pw = float(ARGS.get("elbowsharp", 0.3))
    for s in "lr":
        de = np.linalg.norm(Vo - B.J["lowerarm_" + s], axis=1)
        k_ = np.clip(1 - (de - 6.0) / 6.0, 0, 1)[:, None]               # 1 within 6 cm of the elbow joint, 0 beyond 12
        Ws = Wo ** pw; Ws /= np.maximum(Ws.sum(1, keepdims=True), 1e-9)
        Wo = Wo * (1 - k_) + Ws * k_
    Wo = RG.cleanup(Wo, None, smooth=0)
    W[:n_out] = Wo; W[n_out:2 * n_out] = Wo                       # inner shell = outer: both faces move together
    oT = RG.tri_mesh(Vo, Fo)
    for k, (a, b, r_) in enumerate(M["ranges"][1:], 1):
        P = V[a:b]
        if r_ == G.R_POCKET:                                          # outer half, then the inner half copies it
            half = (b - a) // 2
            W[a:a + half] = RG.nearest_interp(Vo, oT, Wo, P[:half]); W[a + half:b] = W[a:a + half]
        elif r_ == G.R_GROM:
            W[a:b] = RG.nearest_interp(Vo, oT, Wo, P.mean(0, keepdims=True))[0]     # rigid eyelet
        elif r_ == G.R_STRING:
            Ws = RG.nearest_interp(Vo, oT, Wo, P)                    # each cord vertex follows the cloth under it
            ring = 12 if False else 6
            nring = (b - a - 2) // ring
            Wr = Ws[:nring * ring].reshape(nring, ring, -1).mean(1)
            for _ in range(4):
                Wr[1:-1] = 0.25 * Wr[:-2] + 0.5 * Wr[1:-1] + 0.25 * Wr[2:]
            Ws[:nring * ring] = np.repeat(Wr, ring, 0); Ws[nring * ring] = Wr[0]; Ws[nring * ring + 1] = Wr[-1]
            W[a:b] = Ws
            last_cord = (V[a:b], Ws)
        else:                                                         # knot / aglet: follow the cord end, rigid
            cV, cW = last_cord
            d = np.linalg.norm(cV - P.mean(0), axis=1)
            W[a:b] = cW[np.argmin(d)]
    W = RG.cleanup(W, None, smooth=0)
    used = RG.write_groups(low, W)
    RG.bind(low, rig)
    RG.bind(proxy, rig)                                                 # proxy follows the export rig for the QC renders
    nz = (W > 0).sum(1)
    stats = dict(bones=used, max_influences=int(nz.max()), mean_influences=round(float(nz.mean()), 2),
                 proxy_stray_weight_removed=round(float(stray.max()), 3))
    log(f"  hoodie skinned to {len(used)} bones, <= {stats['max_influences']} influences")
    return proxy, W, stats


# ------------------------------------------------------------------------------------------------------------- export
FBX_SK = dict(use_selection=True, apply_unit_scale=True, apply_scale_options="FBX_SCALE_NONE", axis_forward="-Y", axis_up="Z",
              object_types={"ARMATURE", "MESH"}, add_leaf_bones=False, armature_nodetype="NULL", bake_anim=False,
              mesh_smooth_type="FACE", use_tspace=True, path_mode="STRIP", use_armature_deform_only=False,
              use_mesh_modifiers=True, use_custom_props=True, primary_bone_axis="Y", secondary_bone_axis="X")


TRIMS = (G.R_STRING, G.R_KNOT, G.R_AGLET, G.R_GROM)


def _split(o, keep_trims):
    """Copy of o with only the cloth (keep_trims False) or only the trims (True)."""
    c = o.copy(); c.data = o.data.copy()
    for col in o.users_collection:
        col.objects.link(c)
    for m in list(c.modifiers):
        c.modifiers.remove(m)
    bm = bmesh.new(); bm.from_mesh(c.data)
    lay = bm.faces.layers.int.get("reg")
    dead = [f for f in bm.faces if (f[lay] in TRIMS) != keep_trims]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    bm.to_mesh(c.data); bm.free()
    return c


def lods(low):
    """Cloth decimated (collapse) to hit 50 % / 25 % of LOD0 overall; eyelets, cords and aglets stay as built (decimating
    them flattens a cord end into its aglet -> coplanar faces)."""
    out = []
    total = len(low.data.polygons)
    for i, r in enumerate((0.5, 0.25), 1):
        cloth, trims = _split(low, False), _split(low, True)
        nt = len(trims.data.polygons)
        d = cloth.modifiers.new("Decimate", "DECIMATE")
        d.ratio = max(0.05, (r * total - nt) / len(cloth.data.polygons)); d.use_collapse_triangulate = True
        # keep the channel cloth around the eyelets intact (collapsed faces landed on the flanges -> coplanar)
        eyes = [Vector(p.co) for p in trims.data.vertices]
        grp = cloth.vertex_groups.new(name="__lod_keep")
        from mathutils.kdtree import KDTree
        kd = KDTree(len(eyes))
        for k_, e_ in enumerate(eyes):
            kd.insert(e_, k_)
        kd.balance()
        far = [v.index for v in cloth.data.vertices if kd.find(v.co)[2] > 2.5]
        grp.add(far, 1.0, "REPLACE")
        d.vertex_group = "__lod_keep"
        bpy.context.view_layer.objects.active = cloth
        bpy.ops.object.modifier_apply(modifier="Decimate")
        cloth.vertex_groups.remove(cloth.vertex_groups["__lod_keep"])
        bpy.ops.object.select_all(action="DESELECT")
        cloth.select_set(True); trims.select_set(True)
        bpy.context.view_layer.objects.active = cloth
        bpy.ops.object.join()
        o = cloth
        o.name = o.data.name = f"SK_{ITEM}_LOD{i}"
        Wl = RG.cleanup(RG.group_matrix(o, RG.ALLOWED), None, smooth=0)       # decimation blends weights: re-limit to 4
        RG.write_groups(o, Wl)
        RG.bind(o, low.parent)
        o.parent = low.parent
        out.append(o)
    return out


def export_fbx(objs, rig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs + [rig]:
        o.hide_set(False); o.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.fbx(filepath=path, **FBX_SK)
    return path


def reimport_check(paths):
    """Fresh scene: armature + mesh + weights survive; bone names / parents match the skeleton json."""
    want = {n: b["parent"] or "root" for n, b in B.D.items()}; want["root"] = None
    report = {}
    ok_all = True
    for p in paths:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.fbx(filepath=p)
        arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
        meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        r = {"armatures": [a.name for a in arms], "meshes": [m.name for m in meshes]}
        ok = len(arms) == 1 and len(meshes) == 1
        if ok:
            a, m = arms[0], meshes[0]
            got = {b.name: (b.parent.name if b.parent else None) for b in a.data.bones}
            r["bones"] = len(got)
            r["names_match"] = set(got) == set(want)
            r["parents_match"] = all(got.get(n) == pa for n, pa in want.items())
            r["root"] = [b.name for b in a.data.bones if b.parent is None]
            gnames = {g.index: g.name for g in m.vertex_groups}
            sums, counts = [], []
            for v in m.data.vertices:
                ws = [g.weight for g in v.groups if g.weight > 0]
                sums.append(sum(ws)); counts.append(len(ws))
            sums = np.array(sums)
            r["groups"] = sorted(gnames.values())
            r["groups_are_bones"] = set(gnames.values()) <= set(got)
            r["weight_sum_min_max"] = [round(float(sums.min()), 4), round(float(sums.max()), 4)]
            r["max_influences"] = int(max(counts))
            r["tris"] = sum(len(p_.vertices) - 2 for p_ in m.data.polygons)
            r["materials"] = [s.material.name for s in m.material_slots if s.material]
            r["armature_modifier"] = any(md.type == "ARMATURE" and md.object == a for md in m.modifiers)
            ok = (r["names_match"] and r["parents_match"] and r["root"] == ["root"] and r["groups_are_bones"]
                  and abs(r["weight_sum_min_max"][0] - 1) < 1e-3 and abs(r["weight_sum_min_max"][1] - 1) < 1e-3
                  and r["max_influences"] <= 4 and r["armature_modifier"])
        r["ok"] = bool(ok)
        ok_all &= ok
        report[os.path.basename(p)] = r
    print("REIMPORT", "PASS" if ok_all else "FAIL", json.dumps(report))
    return ok_all, report


# ------------------------------------------------------------------------------------------------------------- renders
def cam_at(target, rel, lens=50, name="Cam"):
    cd = bpy.data.cameras.new(name); cd.lens = lens; cd.clip_start = 1; cd.clip_end = 1e5
    c = bpy.data.objects.new(name, cd); bpy.context.scene.collection.objects.link(c)
    c.location = Vector(target) + Vector(rel)
    c.rotation_euler = (Vector(target) - c.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = c
    return c


def label(path, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(os.path.join(ROOT, "fonts", "WorkSans-Bold.ttf"), max(14, im.width // 32))
    d.rectangle((0, 0, im.width, im.width // 32 + 16), fill=(22, 23, 26))
    d.text((10, 6), text, font=f, fill=(235, 235, 235))
    im.save(path)


def clip_metric(low, proxy, rig, n_outer):
    """Per QC pose: hoodie vertices that end up inside the posed proxy body (count, max depth cm)."""
    from mathutils.bvhtree import BVHTree

    def ev(ob):
        dg = bpy.context.evaluated_depsgraph_get(); me = ob.evaluated_get(dg).to_mesh()
        V = [v.co.copy() for v in me.vertices]; F = [list(f.vertices) for f in me.polygons]
        ob.evaluated_get(dg).to_mesh_clear(); return V, F
    out = {}
    for pz, (rots, moves) in RG.poses().items():
        RG.pose(rig, rots, moves)
        pV, pF = ev(proxy); bvh = BVHTree.FromPolygons(pV, pF)
        V, _ = ev(low)
        depth, outer = [], []
        for i, v in enumerate(V):
            loc, nrm, _, d = bvh.find_nearest(v)
            if loc is not None and (v - loc).dot(nrm) < 0:
                depth.append(d)
                if i < n_outer:
                    outer.append(d)
        out[pz] = dict(verts_inside_body=len(depth), max_depth_cm=round(max(depth), 2) if depth else 0.0,
                       outer_surface_verts_inside=len(outer), outer_max_depth_cm=round(max(outer), 2) if outer else 0.0)
    RG.pose(rig, {}, None)
    return out


def fit_colour(low, mats, v="Maroon", target_srgb=(110, 30, 38), iters=6):
    """look_match.py method: render a chest-only crop under the studio light, take the mean colour, scale the layer albedo
    per channel in linear by target/measured, repeat. ColorB (cords, lining) keeps its ratio to ColorA."""
    from PIL import Image
    m = mats[v]
    g = next(n for n in m.node_tree.nodes if n.type == "GROUP")
    low.data.materials[0] = m
    cam_at((0, -10, 125), (0, -60, 0), lens=60, name="fit")
    tgt = tex.srgb_to_lin(np.array(target_srgb) / 255.0)
    ca = np.array(g.inputs["ColorA"].default_value[:3]); cb = np.array(g.inputs["ColorB"].default_value[:3])
    hist = []
    for k in range(iters):
        p = os.path.join(WORK, f"fit_{k}.png")
        scene.render(p, 96, 16)
        im = np.asarray(Image.open(p).convert("RGB"), np.float64)[24:72, 24:72] / 255
        meas = tex.srgb_to_lin(im.reshape(-1, 3).mean(0))
        hist.append([round(float(c) * 255) for c in im.reshape(-1, 3).mean(0)])
        r = np.clip(tgt / np.maximum(meas, 1e-5), 0.2, 4.0)
        ca = np.clip(ca * r, 0, 1); cb = np.clip(cb * r, 0, 1)
        g.inputs["ColorA"].default_value = (*ca, 1); g.inputs["ColorB"].default_value = (*cb, 1)
    FITTED[v] = (tuple(float(c) for c in ca), tuple(float(c) for c in cb)); FIT_HIST[:] = hist
    log(f"  colour fit {v}: rendered sRGB per iteration {hist} -> ColorA {np.round(ca, 4).tolist()}")
    return hist


def renders(low, proxy, rig, mats):
    sc = bpy.context.scene
    skin_m = master.simple("M_ProxySkin", (0.62, 0.30, 0.22), 0.55)
    proxy.data.materials.clear(); proxy.data.materials.append(skin_m)
    low.data.materials[0] = mats["Maroon"]
    for o in bpy.data.collections["Work"].objects:
        o.hide_render = True
    scene.studio([proxy], key=1.0)
    os.makedirs(REND, exist_ok=True)
    fit_colour(low, mats)
    low.data.materials[0] = mats["Maroon"]
    shots = []
    P = RG.poses()
    views = [("rest", (0, -6, 122), (0, -235, 20), "rest A-pose, front"), ("rest", (0, 0, 122), (0, 235, 30), "rest A-pose, back"),
             ("rest", (0, -3, 122), (-150, -180, 40), "rest A-pose, 3/4"),
             ("arms_up_90", (0, 0, 128), (-110, -230, 40), "arms raised 90 deg (clavicle 12 + upperarm)"),
             ("arms_forward", (0, -12, 125), (-175, -150, 50), "arms forward (shoulder flexion ~90 deg)"),
             ("crouch_twist", (0, -6, 104), (-160, -185, 30), "crouch: 60 deg knees, 55 deg hips, 30 deg torso twist"),
             ("elbow_120", (0, -8, 122), (-140, -185, 40), "elbows bent 120 deg")]
    for i, (pz, tgt, rel, txt) in enumerate(views):
        rots, moves = P[pz]
        RG.pose(rig, rots, moves)
        cam_at(tgt, rel, lens=55, name=f"qc{i}")
        p = os.path.join(REND, f"qc_{i}_{pz}.png")
        scene.render(p, 800, min(SMP, 40), res_y=1000)
        label(p, txt)
        shots.append(p)
        log(f"  render {os.path.basename(p)}")
    RG.pose(rig, {}, None)
    # hero: heather grey, 3/4 close
    low.data.materials[0] = mats["HeatherGrey"]
    proxy.data.materials[0] = master.simple("M_ProxyGrey", (0.32, 0.32, 0.33), 0.6)
    cam_at((0, -4, 128), (-120, -170, 25), lens=60, name="hero")
    hero = os.path.join(REND, "hoodie.png")
    scene.render(hero, 1024, SMP, res_y=1024)
    # material close-ups (Maroon): neck / eyelets / cords / seams, and cuff / pocket / band
    from PIL import Image
    low.data.materials[0] = mats["Maroon"]
    cam_at((1, -12, 141), (-16, -42, 6), lens=60, name="detail_a")
    pa = os.path.join(WORK, "detail_a.png"); scene.render(pa, 640, SMP, res_y=640)
    cam_at((22, -12, 104), (-30, -48, 8), lens=60, name="detail_b")
    pb = os.path.join(WORK, "detail_b.png"); scene.render(pb, 640, SMP, res_y=640)
    detail = os.path.join(REND, "hoodie_detail.png")
    dimg = Image.new("RGB", (1280, 640)); dimg.paste(Image.open(pa).convert("RGB"), (0, 0)); dimg.paste(Image.open(pb).convert("RGB"), (640, 0))
    dimg.save(detail)
    low.data.materials[0] = mats["HeatherGrey"]
    log("  hero + detail rendered")
    return shots, hero, detail


def turnaround(low, proxy, mats):
    """The reference's four views (front, back, right profile, 3/4 from the front-right) on the proxy, Maroon, orthographic.
    Our renders are committed; the side-by-side with the user's reference goes to work/ only (the reference is private)."""
    from PIL import Image
    low.data.materials[0] = mats["Maroon"]
    proxy.data.materials[0] = master.simple("M_ProxySkin2", (0.62, 0.30, 0.22), 0.55)
    views = [("front", 0), ("back", 180), ("right", -90), ("threequarter", -30)]
    paths = []
    rig = low.parent
    RG.pose(rig, {"upperarm_l": ((0, 1, 0), 24), "upperarm_r": ((0, 1, 0), -24)})     # arms relaxed like the reference
    sc = bpy.context.scene
    bd = bpy.data.objects.get("Backdrop")
    if bd:
        bd.hide_render = True
    bg = sc.world.node_tree.nodes["Background"].inputs
    bg_old = (tuple(bg["Color"].default_value), bg["Strength"].default_value)
    bg["Color"].default_value = (0.55, 0.56, 0.58, 1); bg["Strength"].default_value = 0.6
    for nm, yaw in views:
        y = math.radians(yaw)
        c = cam_at((0, -2, 92), (400 * math.sin(y), -400 * math.cos(y), 0), name=f"ta_{nm}")
        c.data.type = "ORTHO"; c.data.ortho_scale = 196
        p = os.path.join(WORK, f"turn_{nm}.png")
        scene.render(p, 600, min(SMP, 32), res_y=1086)
        paths.append(p)
    ims = [Image.open(p).convert("RGB") for p in paths]
    RG.pose(rig, {}, None)
    if bd:
        bd.hide_render = False
    bg["Color"].default_value = bg_old[0]; bg["Strength"].default_value = bg_old[1]
    strip = Image.new("RGB", (600 * 4, 1086))
    for i, im in enumerate(ims):
        strip.paste(im, (600 * i, 0))
    strip.save(os.path.join(REND, "hoodie_turnaround.png"))
    ref = os.path.join(HERE, "refs_user", "hoodie_turnaround.webp")
    if os.path.exists(ref):
        r = Image.open(ref).convert("RGB")
        sh = Image.new("RGB", (600 * 4, 1086 * 2), (22, 23, 26))
        for i in range(4):                                       # each reference panel centred over our view
            pnl = r.crop((r.width * i // 4, 0, r.width * (i + 1) // 4, r.height)).resize((r.width // 4 * 1086 // r.height, 1086))
            sh.paste(pnl, (600 * i + (600 - pnl.width) // 2, 0))
        sh.paste(strip, (0, 1086))
        sh.save(os.path.join(WORK, "ref_compare.png"))           # private: work/ is gitignored
    low.data.materials[0] = mats["HeatherGrey"]
    return os.path.join(REND, "hoodie_turnaround.png")


def variant_strip(low, proxy, mats):
    """All colour variants side by side (copies of the hoodie + proxy)."""
    objs = []
    for i, v in enumerate(VARIANTS):
        dx = (i - (len(VARIANTS) - 1) / 2) * 75
        for src in (low, proxy):
            o = src.copy(); o.data = src.data.copy() if src is low else src.data
            bpy.context.scene.collection.objects.link(o)
            o.location.x += dx
            if src is low:
                o.data.materials[0] = mats[v]
            objs.append(o)
    low.hide_render = proxy.hide_render = True
    lo, hi = scene.bounds(objs)
    cam_at(((lo.x + hi.x) / 2, 0, 122), (0, -560, 40), lens=50, name="variants")
    p = os.path.join(REND, "hoodie_variants.png")
    scene.render(p, 1024, min(SMP, 32), res_y=420)
    for o in objs:
        bpy.data.objects.remove(o)
    low.hide_render = proxy.hide_render = False
    return p


def sheet(paths, out, title):
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p).convert("RGB") for p in paths]
    w = 400
    tiles = [im.resize((w, int(im.height * w / im.width))) for im in ims]
    h = max(t.height for t in tiles)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sh = Image.new("RGB", (cols * w + (cols + 1) * 8, rows * (h + 8) + 60), (22, 23, 26))
    d = ImageDraw.Draw(sh)
    d.text((10, 14), title, font=ImageFont.truetype(os.path.join(ROOT, "fonts", "WorkSans-Bold.ttf"), 26), fill=(235, 235, 235))
    for i, t in enumerate(tiles):
        sh.paste(t, (8 + (i % cols) * (w + 8), 52 + (i // cols) * (h + 8)))
    sh.save(out)
    return out


# ------------------------------------------------------------------------------------------------------------- main
def run_qc(blend, objects):
    """tools/mesh_qc.py in this process (opens the saved blend); returns (errors, warnings, summary line, findings)."""
    import runpy, io, contextlib
    bpy.ops.wm.open_mainfile(filepath=blend)
    js = os.path.join(WORK, f"qc_{objects}.json")
    argv = sys.argv
    sys.argv = ["blender", "--", "--objects", objects, "--json", js]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(os.path.join(ROOT, "tools/mesh_qc.py"), run_name="__main__")
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    lines = [l for l in buf.getvalue().splitlines() if l.startswith("QC")]
    for l in lines:
        print(l[:220])
    r = json.load(open(js))
    return r["errors"], r["warnings"], (lines[-1] if lines else ""), r["findings"]


def main():
    scene.reset()
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"; sc.unit_settings.scale_length = 0.01
    lowM, hiM, low, high, L, lay = build_meshes()
    nrm, ao = bakes(low, high, lowM, L)
    texs, lab, R = textures(lowM, L, lay, nrm, ao)
    mats = materials(texs, lab, low)
    rig = B.rig()
    scene.link(rig, scene.collection("Game"))
    proxy, W, wstats = skin(lowM, low, rig)
    scene.link(proxy, scene.collection("Reference"))
    lod = lods(low)
    tris = {o.name: len(o.data.polygons) for o in [low] + lod}
    log(f"tris {tris}")
    # dims (rest), body gap
    lo, hi = scene.bounds([low])
    Vo = lowM["outer"][0]
    thick = G.vertex_thickness(len(Vo), np.asarray(lowM["outer"][1]), np.r_[lowM["body"]["reg"], lowM["hood"]["reg"]])
    gap = B.body_sdf(Vo[:lowM["hood_base"]]) - thick[:lowM["hood_base"]]
    torso = (np.abs(Vo[:lowM["hood_base"], 0]) < 17) & (Vo[:lowM["hood_base"], 2] > 100) & (Vo[:lowM["hood_base"], 2] < 145)
    gaps = dict(min_cm=round(float(gap.min()), 2), torso_p5_p50_p95_cm=[round(float(np.percentile(gap[torso], q)), 2) for q in (5, 50, 95)])
    log(f"body gap (inner face to proxy): {gaps}")
    # export
    paths = [export_fbx([low], rig, f"{FBXD}/SK_{ITEM}.fbx")] + [export_fbx([o], rig, f"{FBXD}/SK_{ITEM}_LOD{i}.fbx") for i, o in enumerate(lod, 1)]
    for o in lod:
        o.hide_render = True
    blend = os.path.join(HERE, "hoodie.blend")
    for o in high:
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    bpy.ops.file.make_paths_relative()
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    shots, hero, detail, vstrip = [], None, None, None
    if RENDERS:
        log("renders")
        shots, hero, detail = renders(low, proxy, rig, mats)
        vstrip = variant_strip(low, proxy, mats)
        turnaround(low, proxy, mats)
        sheet(shots + [hero], os.path.join(REND, "_sheet.png"),
              "SK_Hoodie deformation QC on the proxy body (Maroon) + hero (HeatherGrey)")
    low.data.materials[0] = bpy.data.materials[f"M_{ITEM}"]
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    clip = clip_metric(low, proxy, rig, lowM["n_outer"])
    log(f"clip vs proxy: {clip}")
    lod_names = [o.name for o in lod]
    # gates: FBX re-import (fresh scene), then mesh_qc on the saved blend
    ok, rep = reimport_check(paths)
    qc = {}
    for nm in ["SK_Hoodie"] + lod_names:
        e, w, line, _ = run_qc(blend, nm)
        qc[nm] = dict(errors=e, warnings=w, summary=line)
        log(f"QC {nm}: {line}")
    man = {
        "item": ITEM, "replaces_rows": ["FleeceJacket"], "category": "clothes", "slot": "Shirt", "clip": None, "skeletal": True,
        "real_object": "plain heavy-fleece pullover hoodie, regular fit (user's 4-view reference): hem at the hip (z 89 cm) with a 7 cm rib "
                       "band covering the belt line, slightly dropped set-in shoulders, sleeves to the wrist with 6.8 cm rib cuffs over "
                       "the hand root, kangaroo pocket 34 x 16 cm from the band seam to the navel, lined hood worn down standing "
                       "behind the neck (top ~158 cm, point ~138 cm), 2 metal eyelets ~4 cm off centre, 4 mm round cords ~24 cm with "
                       "metal aglets",
        "dims_cm": [round(hi.x - lo.x, 1), round(hi.y - lo.y, 1), round(hi.z - lo.z, 1)],
        "pivot": "skeletal: origin = root bone at the floor between the feet; Blender cm scene (scale_length 0.01), FBX in cm",
        "skeleton": {"source": "props/brief/skeleton_ue5_manny_rest.json (rig built with anim/pill_bottle/pill_bottle_anim.build_rig)",
                     "bones": 1 + len(B.D), "root": "root (parent of pelvis, non-deforming)",
                     "armature_object": "named 'Armature' so UE does not add the armature node as an extra bone",
                     "bone_rolls": "0 (the json carries positions only) - bind by name onto SK_Mannequin; same caveat as the pill-bottle clip",
                     "skinned_to": wstats["bones"], "max_influences": wstats["max_influences"],
                     "weights": "automatic bone-heat weights (Baran-Popovic: nearest visible bone segment, heat diffusion over the proxy "
                                "surface, scripted with scipy because bpy's heat solver silently dropped pelvis/spine/thigh on the proxy) "
                                "with limb-aligned segments (twist bones own their third of the limb), barycentric transfer to the outer "
                                "cloth surface, caps (thigh 0.45, hand 0.18, head 0.12), 1x edge smoothing, <= 4 influences, crumbs "
                                "< 0.01 pruned, normalised; inner shell copies the outer; "
                                "pocket / eyelets / cords take the weights of the cloth under them (eyelets rigid)"},
        "fit": {"proxy_body": "own SDF mannequin, ~180 cm, A-pose, never exported",
                "gap_inner_face_to_body_cm": gaps,
                "outer_envelope": "relaxed fit: chest/back ~3.7 cm over the body (gap + 0.45 fleece), shoulders ~3 cm, upper sleeve "
                                  "~4.5 cm, snug 7 cm rib band and cuffs 1.25 cm; macro folds +-0.9 cm; hood stands up to ~5 cm proud "
                                  "behind the neck. Vests worn over it should clear body + 4.5 cm on the torso"},
        "tris": tris,
        "textures": {"size": S, "normal": "OpenGL (flip green in UE)",
                     "maps": {k: os.path.basename(v) for k, v in texs.items()},
                     "variants": {v: (f"T_{ITEM}_Label_Print.png" if VARIANTS[v][2] else None) for v in VARIANTS},
                     "texel_density_px_per_cm": round(lay["uv_per_cm"] * S, 1),
                     "density_factors": {G.REGION_NAMES[k]: v for k, v in U.DENSITY.items()}},
        "mask": "R fabric colour A (heathered 0.9-1.0; brushed inside 0.72; stitches lighter), G lining + drawstrings colour B, "
                "B chest print area (NavyPrint variant label), A wear (cuffs, hem, hood edge, pocket openings, elbows, fold ridges)",
        "variant_params": {v: vparams(v) for v in VARIANTS},
        "materials": {"slot": f"M_{ITEM}", "instances": [f"MI_{ITEM}_{v}" for v in VARIANTS]},
        "qc": {"clip_vs_proxy_body": clip, "mesh_qc": qc, "fbx_reimport": "PASS" if ok else "FAIL", "fbx_report": rep},
        "renders": {"sheet": "renders/_sheet.png", "hero": "renders/hoodie.png", "detail": "renders/hoodie_detail.png",
                    "turnaround": "renders/hoodie_turnaround.png (Maroon; front, back, right profile, 3/4 like the user's reference)",
                    "variants": "renders/hoodie_variants.png"},
        "colour_fit": {"Maroon": {"target_srgb": [110, 30, 38], "method": "look_match-style: chest-only crop rendered under "
                              "the studio light, per-channel linear albedo scale target/measured, 6 iterations",
                              "rendered_srgb_per_iteration": FIT_HIST, "ColorA_linear": FITTED.get("Maroon", [None])[0]}},
        "notes": "Sheen: M_Lab_Prop has no sheen input, so the fuzz is faked in the texture set (fold ridges mix 6 % towards the "
                 "lighter substrate via Mask R, roughness 0.9-0.95) and the Blender preview adds Principled sheen 0.3; in UE use "
                 "the Cloth shading model or a fresnel lerp on the MI for real sheen. Knit is not drawn at thread scale (heavy "
                 "fleece reads smooth at viewing distance); rib, cover stitches and side-seam topstitch are in the normal map."}
    with open(os.path.join(HERE, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
