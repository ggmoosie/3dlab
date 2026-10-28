"""Texture synthesis for the triple mag pouch (numpy). Same contract as ../molle_pouch/pouch_tex.py: inputs are bakes of
the game meshes (normal/AO/curvature from the high, part ID, the high's lab_p coordinates, the low's face classes).
  cordura   plain weave + stitch rows (sleeve top binding, gusset top-stitch, tall-back centre + cross line, panel edge)
  webbing   rib weave, bar-tacks (PALS columns, loop tabs, pull tabs, channel bridges), flap perimeter + hook-line stitching
  velcro    loop pile (fuzzy, matte), hook strip (dense stipple); polymer straps moulded, clips/studs/eyelets."""
import os
import numpy as np
from lab import tex
import mag_geo as G
import pouch_tex as PT

NAMES = list(G.IDS)
COLS = np.array([G.IDS[n] for n in NAMES], np.float32)


def classify(idm, hit):
    best = np.zeros(idm.shape[:2], np.int32); bd = np.full(idm.shape[:2], 1e9, np.float32)
    for k, c in enumerate(COLS):
        d = ((idm - c) ** 2).sum(-1); m = d < bd; best[m] = k; bd[m] = d[m]
    return np.where(hit, best, NAMES.index("fabric"))


def core(bk, S, tpm):
    PT.TPM[0] = tpm
    cls = np.rint(bk["cls"]).astype(np.int32)
    cover = cls > 0
    idm, P = bk["id"], bk["p"]
    hit = cover & (idm.sum(-1) > 0.05) & np.isin(cls, (1, 4))
    part = classify(idm, hit)
    F = NAMES.index
    part = np.where(cls == 5, F("webh"), part)
    fab = (part == F("fabric")) & cover
    web = np.isin(part, (F("webh"), F("webv"), F("flapweb")))
    H = np.zeros((S, S), np.float32); thread = np.zeros((S, S), np.float32); cav = np.ones((S, S), np.float32)
    rows, cols = np.mgrid[0:S, 0:S].astype(np.float32)
    u_mm, v_mm = cols / tpm, (S - rows) / tpm
    wv = PT.weave(u_mm, v_mm, max(0.33, 1.5 / tpm), 2)          # ~0.66 mm repeat (real size)
    fz = tex.fbm(S, S, 220, 5, beta=1.2)
    H += fab * (cls != 2) * (0.022 * wv + 0.006 * (fz - 0.5))
    cav *= np.where(fab & (cls != 2), 0.9 + 0.1 * wv, 1)
    Pa, Pb, Pc = P[..., 0], P[..., 1], P[..., 2]
    code = np.rint(Pc).astype(np.int32)

    lines = np.zeros((S, S), np.float32); lth = np.zeros((S, S), np.float32); holes = np.zeros((S, S), np.float32)

    def line(mask, d, along, **kw):
        nonlocal lines, lth, holes
        if not mask.any():
            return
        a, t, h_ = PT.stitch(d[mask], along[mask], **kw)
        lines[mask] += a; lth[mask] = np.maximum(lth[mask], t); holes[mask] = np.maximum(holes[mask], h_)
    tk = np.zeros((S, S), np.float32); tm = np.zeros((S, S), np.float32)

    def tack(mask, dx, across, hw=1.6, hl=11.0):
        nonlocal tk, tm
        if not mask.any():
            return
        a, m = PT.satin(dx[mask], across[mask], hw, hl)
        tk[mask] = np.maximum(tk[mask], a); tm[mask] = np.maximum(tm[mask], m)

    # ---------------------------------------------------------------- cordura: stitch rows from the cube params (x, y, z = Pa, Pb, Pc)
    fh = fab & hit
    x, y, z = Pa, Pb, Pc
    for i, xc in enumerate(G.XC):
        sl = fh & (np.abs(x - xc) <= G.SHX + 0.5) & (y < -0.2) & (z <= G.FRONT_H + 0.5)
        b = G.sleeve_box(i)
        # top binding stitch, 5 mm below the open edge, all round the sleeve
        line(sl & (np.abs(z - (G.FRONT_H - 5)) < 2.5), np.abs(z - (G.FRONT_H - 5)), x + y)
        # gusset top-stitch beside the front seam (sides and bottom)
        side = sl & (np.abs(np.abs(x - xc) - G.SHX) < 0.6)
        line(side & (np.abs(y - (G.SY0 - G.SHY + 4.0)) < 2.5), np.abs(y - (G.SY0 - G.SHY + 4.0)), z)
        bot = sl & (z < 0.6)
        line(bot & (np.abs(y - (G.SY0 - G.SHY + 4.0)) < 2.5), np.abs(y - (G.SY0 - G.SHY + 4.0)), x)
        # tall back (front face, y < 0): centre line and the cross line
        bb = G.back_box(i)
        tb = fh & (np.abs(x - xc) <= bb.h[0] + 0.5) & (z > G.FRONT_H - 30) & (y < bb.c[1] - bb.h[1] + 0.6)
        line(tb & (z > G.FRONT_H + 2) & (z < G.BACK_H - 5) & (np.abs(x - xc) < 2.5), np.abs(x - xc), z)
        line(tb & (np.abs(z - 139.0) < 2.5) & (np.abs(x - xc) < bb.h[0] - 4), np.abs(z - 139.0), x)
        line(tb & (np.abs(z - (G.BACK_H - 4.5)) < 2.5), np.abs(z - (G.BACK_H - 4.5)), x)
    pn = fh & (y > 0.5) & (np.abs(x) > 0) & (y > G.PANEL["y1"] - 0.6)
    p = G.PANEL
    e = np.minimum.reduce([p["hx"] - np.abs(x), z - p["z0"], p["z1"] - z])
    line(pn & (np.abs(e - 4.5) < 2.5), np.abs(e - 4.5), np.where(np.abs(x) > p["hx"] - 6, z, x))

    # ---------------------------------------------------------------- webbing
    wm = web & hit
    rib = 0.5 + 0.5 * np.cos(2 * np.pi * Pa / 0.95)
    weft = 0.5 + 0.5 * np.cos(2 * np.pi * Pb / 0.38)
    Hw = 0.12 * rib ** 0.6 + 0.012 * weft + 0.01 * (fz - 0.5)
    rowsm = wm & (code >= 1) & (code <= 3)
    for t in G.TACKS:
        tack(rowsm, Pa - t, Pb, 1.6, 11.3)
    for j in range(6):                                   # webbing loops above the clips: tack through both layers
        m = wm & (code == 30 + j)
        if m.any():
            smax = Pa[m].max()
            tack(m & (Pa < 7), Pa - 3.0, Pb, 1.5, 9.8)
            tack(m & (Pa > smax - 7), Pa - (smax - 3.0), Pb, 1.5, 9.8)
    for k in range(2):
        m = wm & (code == 40 + k)
        tack(m, np.minimum(np.abs(Pa - 2.5), np.abs(Pa - 15.5)), Pb, 1.4, 5.2)
    for i in range(3):
        m = wm & (code == 70 + i)
        if m.any():
            smax = Pa[m].max()
            tack(m & (Pa < 6), Pa - 2.6, Pb, 1.3, 8.8)
            tack(m & (Pa > smax - 6), Pa - (smax - 2.6), Pb, 1.3, 8.8)
    flap = wm & (code >= 50) & (code <= 52)
    if flap.any():
        for i in range(3):
            m = flap & (code == 50 + i)
            if not m.any():
                continue
            smax = Pa[m].max()
            ins = G.FLAP_W / 2 - 3.0
            line(m & (np.abs(Pb) > ins - 2.5) & (np.abs(Pb) < ins + 2.5), np.abs(np.abs(Pb) - ins), Pa)
            line(m & (Pa > smax - 9), np.abs(Pa - (smax - 4.0)), Pb)
            for off in (24.0, 27.5):                     # the double stitch line above the hook strip
                line(m & (np.abs(Pa - (smax - off)) < 2.5) & (np.abs(Pb) < ins), np.abs(Pa - (smax - off)), Pb)
            line(m & (Pa < 10), np.abs(Pa - 5.0), Pb)
    H += np.where(wm, Hw + tk + lines, 0)
    H += np.where(fh, lines, 0)
    thread = np.maximum(thread, np.maximum(tm, lth))
    cav *= np.where(wm, (0.9 + 0.1 * rib) * (1 - 0.2 * tm), 1)
    cav *= 1 - 0.4 * holes

    # ---------------------------------------------------------------- velcro, polymer, plastics
    loop = hit & (part == F("loop"))
    pile = tex.fbm(S, S, 380, 61, beta=0.9)
    clumps = tex.fbm(S, S, 90, 62, beta=1.5)
    H += loop * (0.18 * pile + 0.12 * clumps)
    cav *= np.where(loop, 0.7 + 0.3 * pile, 1)
    hook = hit & (part == F("hook"))
    H += hook * 0.06 * tex.speckle(S, S, 0.08, 63, 0.7)
    poly = hit & (part == F("poly"))
    H += poly * 0.01 * tex.fbm(S, S, 300, 64)
    plast = hit & (part == F("clip"))
    H += plast * 0.012 * tex.speckle(S, S, 0.02, 9, 0.8)

    detail = PT.grad_normal(H, part * 10 + cls, tpm)
    base = np.where((hit & (cls != 5))[..., None], bk["nrm"], np.array([0.5, 0.5, 1.0], np.float32))
    nrm = tex.blend_normals(base, detail)
    ao = np.where(hit, bk["ao"], np.where(cls == 2, 0.45, np.where(cls == 3, 0.8, 0.5))).astype(np.float32)
    ao = np.clip(ao * cav, 0, 1)
    curv = np.where(hit, bk["curv"], 0.5)
    rough = np.full((S, S), 0.93, np.float32); metal = np.zeros((S, S), np.float32)
    basec = np.zeros((S, S, 3), np.float32) + tex.srgb_to_lin(np.array([0.62, 0.60, 0.56]))
    R = np.where(fab, (1 - 0.05 * wv * (cls != 2)) * (1 - lth), 0).astype(np.float32)
    Gm = np.where(fab, lth, 0).astype(np.float32)
    Gm = np.where(web | (cls == 5), 1 - 0.05 * rib, Gm)
    Bm = np.where(fab, R, 0) + np.where(web, 1 - np.maximum(tm, lth) * 0.9, 0)
    rough = np.where(web, 0.82, rough)
    hard = {"loop": (0.85, 0.97, 0.0, (0.05, 0.05, 0.05)), "hook": (0.85, 0.7, 0.0, (0.04, 0.04, 0.04)),
            "poly": (0.35, 0.48, 0.0, (0.025, 0.025, 0.027)), "clip": (0.7, 0.5, 0.0, (0.03, 0.03, 0.03)),
            "stud": (0.0, 0.32, 1.0, (0.09, 0.09, 0.1)), "eyelet": (0.0, 0.32, 1.0, (0.09, 0.09, 0.1)),
            "mag": (0.0, 0.5, 0.15, (0.04, 0.04, 0.045))}
    for n, (g, r_, m_, c) in hard.items():
        k = part == F(n)
        Gm = np.where(k, g, Gm); rough = np.where(k, r_, rough); metal = np.where(k, m_, metal)
        basec = np.where(k[..., None], np.array(c, np.float32), basec)
        R = np.where(k, 0, R); Bm = np.where(k, 0, Bm)
    Bm = np.where(loop, 0.85, Bm)
    convex = np.clip((curv - 0.70) / 0.14, 0, 1)
    nz = tex.fbm(S, S, 30, 21)
    wear = np.clip(convex * (0.75 + 0.35 * nz), 0, 1) * np.where(fab, 1.0, 0.8)
    wear = np.where(np.isin(part, (F("clip"), F("poly"), F("mag"))), np.clip(convex * (0.6 + 0.5 * nz), 0, 1), wear)
    wear = np.where(cls == 2, 0, wear)
    return [basec, nrm, ao, rough, metal, R, Gm, Bm, np.clip(wear, 0, 1)]


def build(bk, S, tpm, outdir, item):
    return PT.build(bk, S, tpm, outdir, item, core_fn=core)
