"""Texture synthesis for the 6x6 pouch (numpy). Input: per-texel bakes of the game mesh (high->low normal, AO, curvature,
part ID, the high's 'lab_p' coordinates, the low's face classes). Output: T_<Item>_BaseColor / Normal / ORM / Mask and a
camo print label. All detail is placed in real millimetres:
  fabric   500D/1000D plain weave (texel space, uniform density), stitch rows along every seam from the high's cube params
  webbing  ribs across the length, selvage, satin bar-tacks at every channel, box-X / perimeter stitching on straps & tabs
  cord     paracord diamond braid from (arc length, angle) around the tube
  plastic  molded stipple; metal smooth.  Fuzz: fibre noise in the normal, lighter yarn crowns in the mask (sheen)."""
import math, os
import numpy as np
from scipy import ndimage
from lab import tex
import pouch_geo as G

NAMES = list(G.IDS)
COLS = np.array([G.IDS[n] for n in NAMES], np.float32)


def classify(idm, hit):
    best = np.zeros(idm.shape[:2], np.int32)
    bd = np.full(idm.shape[:2], 1e9, np.float32)
    for k, c in enumerate(COLS):
        d = ((idm - c) ** 2).sum(-1)
        m = d < bd
        best[m] = k; bd[m] = d[m]
    return np.where(hit, best, NAMES.index("fabric"))


TPM = [4.5]                      # texels per mm of the arrays being built (set by core)


def stitch(d, along, L=2.35, gap=0.65, w=0.40, h=0.17):
    """lock-stitch row: crisp anti-aliased thread dashes (pitch L+gap mm, thread w mm), needle holes, slight pucker"""
    k = TPM[0]
    per = L + gap
    ph = np.mod(along, per)
    end = np.minimum(ph, L - ph)                                     # mm to the nearer dash end (negative in the gap)
    dash = np.clip(end * k + 0.5, 0, 1)
    cov = np.clip((w / 2 - d) * k + 0.5, 0, 1)                       # pixel-coverage box across the thread
    thread = cov * dash
    round_ = np.sqrt(np.clip(1 - (d / (w / 2 + 1e-6)) ** 2, 0, 1)) * np.clip(np.sin(np.pi * np.clip(ph / L, 0, 1)) * 3, 0, 1)
    hh = h * thread * (0.55 + 0.45 * round_)
    hole = (ph > L) * np.clip((0.22 - np.hypot(d, ph - L - gap / 2)) * k + 0.5, 0, 1)
    hh += -0.05 * np.exp(-(d / 0.9) ** 2) - 0.08 * hole
    return hh, thread, hole


def satin(dx, across, half_w=1.7, half_l=11.5, pitch=0.42):
    """bar-tack: dense zig-zag thread column (dx across the column, `across` along it)"""
    inside = (np.abs(dx) < half_w) & (np.abs(across) < half_l)
    taper = np.clip((half_w - np.abs(dx)) / 0.35, 0, 1) * np.clip((half_l - np.abs(across)) / 0.4, 0, 1)
    z = 0.5 + 0.5 * np.cos(2 * np.pi * (across + 0.32 * np.abs(dx)) / pitch)
    m = inside * taper
    return m * (0.42 + 0.16 * z ** 0.7), m


def weave(u, v, pitch=1.0, seed=0):
    """plain weave height 0..1 (u, v in mm); yarn thickness jitter per yarn"""
    a, b = u / pitch, v / pitch
    ia, ib = np.floor(a), np.floor(b)
    fa, fb = a - ia, b - ib
    r = np.random.default_rng(seed).uniform(0.85, 1.15, 4096).astype(np.float32)
    ja = r[(ia.astype(np.int64)) % 4096]; jb = r[(ib.astype(np.int64) * 7 + 13) % 4096]
    warp = np.sin(np.pi * fa) ** 0.75 * (0.55 + 0.45 * np.cos(np.pi * (b + ia))) * ja
    weft = np.sin(np.pi * fb) ** 0.75 * (0.55 - 0.45 * np.cos(np.pi * (a + ib))) * jb
    return np.clip(np.maximum(warp, weft), 0, 1.2) / 1.2


def grad_normal(H, part, strength):
    """height (mm) -> OpenGL normal, differences only between texels of the same part (no seams between parts)"""
    def d(axis, sgn):
        Hn = np.roll(H, -sgn, axis); Pn = np.roll(part, -sgn, axis)
        return np.where(Pn == part, Hn, H)
    hx1, hx0 = d(1, 1), d(1, -1)
    hy1, hy0 = d(0, -1), d(0, 1)          # row 0 = v 1: +v is one row up
    dx = (hx1 - hx0) * 0.5 * strength
    dy = (hy1 - hy0) * 0.5 * strength
    n = np.stack([-dx, -dy, np.ones_like(H)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def dilate(arrs, cover):
    idx = ndimage.distance_transform_edt(~cover, return_distances=False, return_indices=True)
    return [a[idx[0], idx[1]] for a in arrs]


def strap_z_of_s(i):
    P = G.strap_path(G.STRAP_X[i], True)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return s, P[:, 2]


def core(bk, S, tpm):
    TPM[0] = tpm
    cls = np.rint(bk["cls"]).astype(np.int32)            # 0 none 1 shell 2 liner 3 rim 4 part 5 hidden
    cover = cls > 0
    idm, P = bk["id"], bk["p"]
    hit = cover & (idm.sum(-1) > 0.05) & np.isin(cls, (1, 4))
    part = classify(idm, hit)
    part = np.where(cls == 5, NAMES.index("webh"), part)
    F = NAMES.index
    fab = (part == F("fabric")) & cover
    web = np.isin(part, (F("webh"), F("webv")))
    H = np.zeros((S, S), np.float32)
    thread = np.zeros((S, S), np.float32)
    cav = np.ones((S, S), np.float32)
    rows, cols = np.mgrid[0:S, 0:S].astype(np.float32)
    u_mm, v_mm = cols / tpm, (S - rows) / tpm

    # ---------------------------------------------------------------- fabric: weave + fuzz everywhere, stitches on hits
    wv = weave(u_mm, v_mm, 3.0 / tpm, 1)                   # ~0.66 mm over/under repeat = 3 px at the final 2K
    fz = tex.fbm(S, S, 220, 5, beta=1.2)                  # fibre fuzz
    H += fab * (cls != 2) * (0.04 * wv + 0.006 * (fz - 0.5))
    H += fab * (cls == 2) * 0.01 * (fz - 0.5)
    cav *= np.where(fab, 0.84 + 0.16 * wv, 1)
    bh = fab & hit
    C = P[bh]
    lab, dF, dB, dZ, dS = G.seam_dists(C)
    x, y, z = C[:, 0], C[:, 1], C[:, 2]
    side = (lab == 2) | (lab == 3)
    along_edge = np.where(side, z, x)
    hh = np.zeros(len(C)); th = np.zeros(len(C)); ho = np.zeros(len(C))

    def acc(mask, d, along, **kw):
        nonlocal hh, th, ho
        a, t, h_ = stitch(d, along, **kw)
        hh += np.where(mask, a, 0); th = np.maximum(th, np.where(mask, t, 0)); ho = np.maximum(ho, np.where(mask, h_, 0))
    acc((lab != 0) & (dF < 9), np.abs(dF - 4.5), along_edge)                      # gusset top-stitch by the front seam
    acc((lab != 1) & (dB < 9), np.abs(dB - 4.5), along_edge)                      # ... and by the back seam
    zz = (side & (z > G.ZCUT + 2)) | (lab == 4)
    for off in (9.0, 12.5):                                                       # double rows each side of the zipper
        acc(zz & (dZ < 16), np.abs(dZ - off), along_edge)
    acc(side & (np.abs(z - (G.ZCUT - 3.5)) < 3), np.abs(z - (G.ZCUT - 3.5)), y)   # side seam
    acc(side & (np.abs(z - (G.ZCUT + 3.5)) < 3) & (np.abs(y) > 13), np.abs(z - (G.ZCUT + 3.5)), y)
    sa, sm = satin(z - (G.ZCUT + 2.6), y, 1.9, 6.5)                               # zipper-end bar-tack
    hh += np.where(side, sa, 0); th = np.maximum(th, np.where(side, sm, 0))
    Hf = np.zeros((S, S), np.float32); Hf[bh] = hh; H += Hf
    Tf = np.zeros((S, S), np.float32); Tf[bh] = th; thread = np.maximum(thread, Tf)
    Of = np.zeros((S, S), np.float32); Of[bh] = ho; cav *= 1 - 0.45 * Of
    cav *= np.where(bh, 1 - 0.25 * np.clip(-Hf, 0, 0.2) / 0.2, 1)

    # ---------------------------------------------------------------- webbing
    wm = web & hit
    Pa, Pb, code = P[..., 0], P[..., 1], np.rint(P[..., 2]).astype(np.int32)
    rib = 0.5 + 0.5 * np.cos(2 * np.pi * Pa / 0.95)
    weft = 0.5 + 0.5 * np.cos(2 * np.pi * Pb / 0.38)
    sel = np.clip((np.abs(Pb) - (G.WEB / 2 - 0.9)) / 0.5, 0, 1)
    Hw = 0.12 * rib ** 0.6 + 0.012 * weft + 0.03 * sel + 0.01 * (fz - 0.5)
    tk = np.zeros((S, S), np.float32); tm = np.zeros((S, S), np.float32)

    def tack(mask, dx, across, hw=1.7, hl=11.6):
        nonlocal tk, tm
        a, m = satin(dx, across, hw, hl)
        tk = np.where(mask, np.maximum(tk, a), tk); tm = np.where(mask, np.maximum(tm, m), tm)
    front = wm & (code >= 1) & (code <= 3)
    for t in G.TACK_F:
        tack(front, Pa - t * (1 - 0.025 * (abs(t) > 70)), Pb)
    grid = wm & (code >= 4) & (code <= 6)
    hw_grid = (G.GRID_Z[1] - G.GRID_Z[0]) / 6 - 0.4
    for t in G.TACK_B:
        tack(grid, Pa - t, Pb, 1.7, hw_grid)
    # straps: box-X anchor at the bottom, box stitch through the folded top loop
    lines = np.zeros((S, S), np.float32); lth = np.zeros((S, S), np.float32)

    def line(mask, d, along):
        nonlocal lines, lth
        a, t, h_ = stitch(d, along)
        lines = np.where(mask, lines + a, lines); lth = np.where(mask, np.maximum(lth, t), lth)
    ins = G.WEB / 2 - 2.4
    for i in range(2):
        m = wm & (code == 7 + i)
        s_arr, z_arr = strap_z_of_s(i)
        zs = np.interp(Pa, s_arr, z_arr)
        box = m & (Pa > 1.5) & (Pa < 17.5)
        line(box & (np.abs(Pb) < ins + 1), np.abs(np.abs(Pb) - ins), Pa)
        line(box & (np.abs(Pb) < ins + 1), np.minimum(np.abs(Pa - 2.5), np.abs(Pa - 16.5)), Pb)
        dg = np.minimum(np.abs(Pb - (Pa - 9.5) * ins / 7.0), np.abs(Pb + (Pa - 9.5) * ins / 7.0)) / 1.35
        line(box & (np.abs(Pb) < ins) & (Pa > 2.5) & (Pa < 16.5), dg, Pa * 1.35)
        top = m & (zs > 147.5) & (zs < 157.5)
        line(top & (np.abs(Pb) < ins + 1), np.abs(np.abs(Pb) - ins), zs)
        line(top & (np.abs(Pb) < ins + 1), np.minimum(np.abs(zs - 148.6), np.abs(zs - 156.4)), Pb)
    for i in range(2):                                    # snap tabs: perimeter stitch
        m = wm & (code == 9 + i)
        ins_t = (G.WEB + 0.6) / 2 - 2.2
        rect = m & (Pa > 2.5) & (Pa < 25.5) & (np.abs(Pb) < ins_t + 1)
        line(rect, np.abs(np.abs(Pb) - ins_t), Pa)
        line(rect, np.minimum(np.abs(Pa - 3.5), np.abs(Pa - 24.5)), Pb)
    for i in range(2):                                    # side loops: bar-tack across the top, both layers
        m = wm & (code == 11 + i)
        smax = np.where(m, Pa, 0).max() if m.any() else 0
        tack(m & (Pa < 8), Pa - 3.0, Pb, 1.6, 11.2)
        tack(m & (Pa > smax - 8), Pa - (smax - 2.6), Pb, 1.6, 11.2)
    H += np.where(wm, Hw + tk + lines, 0)
    H += np.where(web & ~hit, 0.04 * rib, 0)
    thread = np.maximum(thread, np.maximum(tm, lth))
    tedge = np.clip(tm * 4, 0, 1) * (1 - np.clip(tm * 1.2, 0, 1))       # dark rim where the tack pulls the webbing in
    cav *= np.where(wm, (0.84 + 0.16 * rib) * (1 - 0.35 * tm * (1 - (0.5 + 0.5 * np.cos(2 * np.pi * Pb / 0.42)))) * (1 - 0.3 * tedge), 1)

    # ---------------------------------------------------------------- tape, cord, plastics
    tape = hit & (part == F("tape"))
    H += tape * 0.03 * (0.5 + 0.5 * np.cos(2 * np.pi * Pa / 0.55))
    cord = hit & (part == F("cord"))
    c1 = 0.5 + 0.5 * np.cos(2 * np.pi * Pa / 1.7 + 8 * Pb)
    c2 = 0.5 + 0.5 * np.cos(2 * np.pi * Pa / 1.7 - 8 * Pb)
    br = np.maximum(c1, c2)
    H += cord * 0.12 * br ** 0.7
    cav *= np.where(cord, 0.82 + 0.18 * br, 1)
    plast = hit & np.isin(part, (F("cap"), F("stiff"), F("teeth")))
    stip = tex.speckle(S, S, 0.02, 9, 0.8)
    H += plast * 0.012 * stip

    # ---------------------------------------------------------------- normal
    detail = grad_normal(H, part * 10 + cls, tpm)
    base = np.where((hit & (cls != 5))[..., None], bk["nrm"], np.array([0.5, 0.5, 1.0], np.float32))
    nrm = tex.blend_normals(base, detail)

    # ---------------------------------------------------------------- ORM / mask / substrate
    ao = np.where(hit, bk["ao"], np.where(cls == 2, 0.55, np.where(cls == 3, 0.8, 0.5))).astype(np.float32)
    ao = np.clip(ao * cav, 0, 1)
    curv = np.where(hit, bk["curv"], 0.5)
    rough = np.full((S, S), 0.93, np.float32)
    metal = np.zeros((S, S), np.float32)
    basec = np.zeros((S, S, 3), np.float32) + tex.srgb_to_lin(np.array([0.62, 0.60, 0.56]))   # abraded nylon fuzz
    R = np.zeros((S, S), np.float32); Gm = np.zeros((S, S), np.float32); Bm = np.zeros((S, S), np.float32)
    R = np.where(fab, (1 - 0.10 * wv * (cls != 2)) * (1 - 0.55 * thread), R)
    Gm = np.where(fab, 0.55 * thread, Gm)
    Gm = np.where(web | (cls == 5), 1 - 0.05 * rib, Gm)
    Bm = np.where(fab, R, 0) + np.where(web, 1 - np.maximum(tm, lth) * 0.9, 0)
    rough = np.where(web, 0.82, rough)
    hard = {"tape": (1.0, 0.8, 0.0, (0.3, 0.3, 0.3)), "teeth": (1.0, 0.42, 0.0, (0.03, 0.03, 0.03)),
            "slider": (1.0, 0.32, 1.0, (0.55, 0.55, 0.56)), "cord": (1.0, 0.85, 0.0, (0.3, 0.3, 0.3)),
            "cap": (0.92, 0.5, 0.0, (0.03, 0.03, 0.03)), "stiff": (0.92, 0.5, 0.0, (0.03, 0.03, 0.03)),
            "stud": (0.0, 0.3, 1.0, (0.10, 0.10, 0.11))}
    for n, (g, r_, m_, c) in hard.items():
        k = part == F(n)
        Gm = np.where(k, g, Gm); rough = np.where(k, r_, rough); metal = np.where(k, m_, metal)
        basec = np.where(k[..., None], np.array(c, np.float32), basec)
        R = np.where(k, 0, R)
    Bm = np.where(part == F("tape"), 1, Bm)
    convex = np.clip((curv - 0.70) / 0.14, 0, 1)
    nz = tex.fbm(S, S, 30, 21)
    wear = np.clip(convex * (0.75 + 0.35 * nz), 0, 1) * np.where(fab, 1.0, 0.8)   # edges and corners only
    wear = np.where(np.isin(part, (F("slider"), F("cap"), F("stiff"))), np.clip(convex * (0.6 + 0.5 * nz), 0, 1), wear)
    wear = np.where(cls == 2, 0, wear)
    return [basec, nrm, ao, rough, metal, R, Gm, Bm, np.clip(wear, 0, 1)]


SS = 2


def upsample(bk, S):
    """2x: dilate the bakes into the gutters, then bilinear (coordinates, normals, AO) / nearest (classes, IDs)"""
    cover = np.rint(bk["cls"]) > 0
    out = {}
    for k, v in bk.items():
        if k in ("cls", "id"):
            out[k] = np.repeat(np.repeat(v, SS, 0), SS, 1)
            continue
        v = dilate([v], cover)[0]
        if v.ndim == 2:
            out[k] = ndimage.zoom(v, SS, order=1, mode="nearest")
        else:
            out[k] = np.stack([ndimage.zoom(v[..., c], SS, order=1, mode="nearest") for c in range(v.shape[2])], -1)
    return out


def down(a):
    h, w = a.shape[:2]
    return a.reshape(h // SS, SS, w // SS, SS, *a.shape[2:]).mean((1, 3))


def finish(maps, bk, S, tpm, outdir, item, camo_fn=None):
    """2x maps -> box-filtered to S, dilated into the gutters, saved"""
    cover = np.rint(bk["cls"]) > 0
    basec, nrm, ao, rough, metal, R, Gm, Bm, wear = [down(m) for m in maps]
    n = nrm * 2 - 1
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    nrm = n * 0.5 + 0.5
    basec, nrm, ao, rough, metal, R, Gm, Bm, wear = dilate([basec, nrm, ao, rough, metal, R, Gm, Bm, wear], cover)
    os.makedirs(outdir, exist_ok=True)
    p = lambda k: os.path.join(outdir, f"T_{item}_{k}.png")
    tex.save(tex.lin_to_srgb(basec), p("BaseColor"))
    tex.save(nrm, p("Normal"))
    tex.save(np.stack([ao, rough, metal], -1), p("ORM"))
    tex.save(np.stack([R, Gm, Bm, wear], -1), p("Mask"), "RGBA")
    lab = camo(S, tpm)
    lp = tex.save(np.concatenate([tex.lin_to_srgb(lab), np.ones((S, S, 1), np.float32)], -1), p("Label_MultiTerrain"), "RGBA")
    return {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}, lp


def build(bk, S, tpm, outdir, item, core_fn=None):
    hi = upsample(bk, S)
    maps = (core_fn or core)(hi, S * SS, tpm * SS)
    del hi
    return finish(maps, bk, S, tpm, outdir, item)


def camo(S, tpm):
    """original multi-terrain style camo (linear RGB): soft khaki/green field, brown + green blobs, dark branches, cream"""
    cyc = S / tpm / 1.0                                  # image width in mm
    f = lambda mm: cyc / mm                               # cycles per image for a feature size in mm
    c = lambda h: np.array(tex.srgb_to_lin(tex.hex2rgb(h)), np.float32)
    def smooth(n, t, w):
        return np.clip((n - t) / w + 0.5, 0, 1)[..., None]
    g = tex.spectral(S, S, f(400), f(120), 2.5, 41)
    out = c("#b5a47e") * (1 - smooth(g, 0.5, 0.35)) + c("#959470") * smooth(g, 0.5, 0.35)
    n1 = tex.spectral(S, S, f(160), f(25), 2.4, 42)
    out = out * (1 - smooth(n1, 0.62, 0.06) * 0.85) + c("#7d6649") * smooth(n1, 0.62, 0.06) * 0.85
    n2 = tex.spectral(S, S, f(140), f(20), 2.4, 43)
    out = out * (1 - smooth(n2, 0.64, 0.05) * 0.9) + c("#5b6b43") * smooth(n2, 0.64, 0.05) * 0.9
    n3 = tex.spectral(S, S, f(60), f(6), 1.9, 44, ax=0.45)
    br = smooth(n3, 0.70, 0.02)
    out = out * (1 - br) + c("#4b3a2b") * br
    n4 = tex.spectral(S, S, f(50), f(6), 2.0, 45)
    cr = smooth(n4, 0.73, 0.03) * 0.9
    out = out * (1 - cr) + c("#d2c7a4") * cr
    return out
