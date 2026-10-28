"""softgoods/core.py - texture-space modelling for soft gear (carriers, pouches, belts, packs). Pure numpy, no IO.
Every item is a set of flat PANELS: an outline in panel mm (u along the panel / its path, v across) plus features
(PALS webbing, loop field, binding, stitch rows, bar-tacks, seams, wrinkles, pockets, blank labels). The SAME panel drives
the low-poly mesh (panel.py maps it onto the body) and these height/ID layers, so geometry and texture always agree.
Layers per panel: H (mm height), ID (material id), thread (stitch coverage), cover (inside the outline)."""
import numpy as np
from scipy import ndimage

FABRIC, WEBBING, LOOP, BINDING, THREAD, ELASTIC, LABEL = range(7)
#            name        albedo x  rough  weave pitch  weave amp
MATS = {FABRIC:  ("cordura", 1.00, 0.88, 0.55, 0.05),
        WEBBING: ("webbing", 0.92, 0.80, 0.45, 0.06),
        LOOP:    ("loop",    1.06, 0.95, 0.00, 0.00),
        BINDING: ("binding", 0.88, 0.80, 0.45, 0.05),
        THREAD:  ("thread",  0.70, 0.70, 0.00, 0.00),
        ELASTIC: ("elastic", 0.85, 0.85, 0.35, 0.08),
        LABEL:   ("label",   1.25, 0.75, 0.25, 0.03)}


# ------------------------------------------------------------------ geometry helpers (mm)
def seg_field(U, V, pts, closed=True):
    """distance to a polyline + arc-length parameter of the nearest point (+ even-odd inside if closed)"""
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P, P[:1]])
    d = np.full(U.shape, 1e9, np.float32); s = np.zeros(U.shape, np.float32); acc = 0.0
    inside = np.zeros(U.shape, bool)
    for a, b in zip(P[:-1], P[1:]):
        e = b - a; L = float(np.hypot(*e)) + 1e-9
        t = np.clip(((U - a[0]) * e[0] + (V - a[1]) * e[1]) / L ** 2, 0, 1)
        dd = np.hypot(U - a[0] - e[0] * t, V - a[1] - e[1] * t)
        m = dd < d; d = np.where(m, dd, d); s = np.where(m, acc + t * L, s); acc += L
        if closed:
            inside ^= ((a[1] > V) != (b[1] > V)) & (U < (b[0] - a[0]) * (V - a[1]) / (b[1] - a[1] + 1e-12) + a[0])
    return (np.where(inside, -d, d) if closed else d), s


def rrect(u0, v0, u1, v1, r=6.0, n=6):
    """rounded rectangle outline"""
    r = min(r, (u1 - u0) / 2, (v1 - v0) / 2); pts = []
    for cx, cy, a0 in ((u1 - r, v0 + r, -90), (u1 - r, v1 - r, 0), (u0 + r, v1 - r, 90), (u0 + r, v0 + r, 180)):
        for k in range(n + 1):
            a = np.radians(a0 + 90 * k / n); pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return pts


def offset_poly(pts, off):
    """inset (off > 0) a convex-ish outline along vertex normals (good enough for stitch rows)"""
    P = np.asarray(pts, float); n = len(P)
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]); sg = 1 if area > 0 else -1
    out = []
    for i in range(n):
        a, b, c = P[i - 1], P[i], P[(i + 1) % n]
        n1 = np.array([-(b - a)[1], (b - a)[0]]); n2 = np.array([-(c - b)[1], (c - b)[0]])
        n1 /= np.linalg.norm(n1) + 1e-9; n2 /= np.linalg.norm(n2) + 1e-9
        m = n1 + n2; m /= np.linalg.norm(m) + 1e-9
        out.append(b + sg * m * off / max(np.dot(m, n1), 0.4))
    return out


# ------------------------------------------------------------------ micro patterns (from the GP pouch, generalised)
def stitch(d, along, tpm, L=2.6, gap=0.8, w=0.45, h=0.18):
    per = L + gap; ph = np.mod(along, per)
    dash = np.clip(np.minimum(ph, L - ph) * tpm + 0.5, 0, 1)
    thread = np.clip((w / 2 - d) * tpm + 0.5, 0, 1) * dash
    hole = (ph > L) * np.clip((0.25 - np.hypot(d, ph - L - gap / 2)) * tpm + 0.5, 0, 1)
    return h * thread - 0.06 * np.exp(-(d / 1.0) ** 2) - 0.08 * hole, thread


def satin(dx, across, half_w=1.8, half_l=11.0, pitch=0.45):
    inside = (np.abs(dx) < half_w) & (np.abs(across) < half_l)
    taper = np.clip((half_w - np.abs(dx)) / 0.35, 0, 1) * np.clip((half_l - np.abs(across)) / 0.4, 0, 1)
    z = 0.5 + 0.5 * np.cos(2 * np.pi * (across + 0.32 * np.abs(dx)) / pitch)
    m = inside * taper
    return m * (0.40 + 0.18 * z ** 0.7), m


def weave(U, V, pitch, seed=0):
    a, b = U / pitch, V / pitch; ia, ib = np.floor(a), np.floor(b); fa, fb = a - ia, b - ib
    r = np.random.default_rng(seed).uniform(0.85, 1.15, 4096).astype(np.float32)
    ja = r[ia.astype(np.int64) % 4096]; jb = r[(ib.astype(np.int64) * 7 + 13) % 4096]
    warp = np.sin(np.pi * fa) ** 0.75 * (0.55 + 0.45 * np.cos(np.pi * (b + ia))) * ja
    weft = np.sin(np.pi * fb) ** 0.75 * (0.55 - 0.45 * np.cos(np.pi * (a + ib))) * jb
    return np.clip(np.maximum(warp, weft), 0, 1.2) / 1.2


def noise(shape, sigma_px, seed):
    n = ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal(shape).astype(np.float32), sigma_px)
    return n / (n.std() + 1e-9)


# ------------------------------------------------------------------ features
def _rect_uv(U, V, r):
    u0, v0, u1, v1 = r
    return (U >= u0) & (U <= u1) & (V >= v0) & (V <= v1)


def f_molle(L, U, V, tpm, rect, rows=None, pitch=38.1, web=25.4, seed=0, **_):
    """PALS: 1 in webbing rows on 1 in gaps, bar-tacked every 1.5 in; stitch rows along each strip edge"""
    u0, v0, u1, v1 = rect
    n = rows or max(1, int((v1 - v0 + 2) // (2 * web)))
    ncol = max(1, int(round((u1 - u0) / pitch)))
    uc0 = (u0 + u1) / 2 - ncol * pitch / 2
    for i in range(n):
        vc = v1 - web / 2 - i * 2 * web
        dv = V - vc; inside = (np.abs(dv) < web / 2) & (U > u0) & (U < u1)
        edge = np.clip((web / 2 - np.abs(dv)) / 0.8, 0, 1)
        rib = 0.05 * (0.5 + 0.5 * np.sin(2 * np.pi * U / 0.9))
        h = inside * (1.3 * edge ** 0.5 + rib)
        L["H"] = np.where(inside, np.maximum(L["H"], L["base"] + h), L["H"]); L["ID"][inside] = WEBBING
        for k in range(ncol + 1):
            uk = uc0 + k * pitch
            hb, m = satin(U - uk, dv, half_l=web / 2 - 1.2)
            sel = (m > 0.02) & inside
            L["H"] = np.where(sel, L["H"] + hb, L["H"]); L["thread"] = np.maximum(L["thread"], m * inside)
        for ve in (vc - web / 2 + 1.6, vc + web / 2 - 1.6):
            hs, th = stitch(np.abs(V - ve), U, tpm, h=0.12)
            sel = inside & (np.abs(V - ve) < 1.5)
            L["H"] = np.where(sel, L["H"] + hs, L["H"]); L["thread"] = np.maximum(L["thread"], th * sel)


def f_loop(L, U, V, tpm, rect, seed=0, **_):
    """hook-and-loop field: raised fuzzy pile, sewn round its edge"""
    d, s = seg_field(U, V, rrect(*rect, r=3)); inside = d < 0
    fuzz = 0.22 * np.clip(noise(U.shape, 0.6, seed) * 0.5 + 0.5, 0, 1)
    L["H"] = np.where(inside, L["base"] + 0.9 * np.clip(-d / 1.0, 0, 1) + fuzz, L["H"]); L["ID"][inside] = LOOP
    hs, th = stitch(np.abs(d + 2.0), s, tpm); sel = np.abs(d + 2.0) < 1.5
    L["H"] = np.where(sel, L["H"] + hs, L["H"]); L["thread"] = np.maximum(L["thread"], th * sel)


def f_label(L, U, V, tpm, rect, **_):
    """blank woven tag (no text, no logo)"""
    d, s = seg_field(U, V, rrect(*rect, r=1.5)); inside = d < 0
    L["H"] = np.where(inside, L["H"] + 0.35, L["H"]); L["ID"][inside] = LABEL


def f_stitch(L, U, V, tpm, pts, closed=False, **_):
    d, s = seg_field(U, V, pts, closed=closed); d = np.abs(d); sel = d < 1.5
    hs, th = stitch(d, s, tpm)
    L["H"] = np.where(sel, L["H"] + hs, L["H"]); L["thread"] = np.maximum(L["thread"], th * sel)


def f_seam(L, U, V, tpm, pts, depth=0.8, width=2.5, **_):
    """sewn seam / fold: a soft valley with a stitch row beside it"""
    d, s = seg_field(U, V, pts, closed=False)
    L["H"] = L["H"] - depth * np.exp(-(d / width) ** 2)
    f_stitch(L, U + 0, V, tpm, [(p[0], p[1] - 3.0) for p in pts])


def f_wrinkle(L, U, V, tpm, rect, angle=0.0, amp=0.9, wave=14.0, seed=0, **_):
    """compression / tension wrinkles: warped waves along `angle`, fading out of the rect"""
    u0, v0, u1, v1 = rect; a = np.radians(angle)
    q = U * np.cos(a) + V * np.sin(a); p = -U * np.sin(a) + V * np.cos(a)
    warp = noise(U.shape, 45 * tpm, seed) * wave * 0.35                        # long, gently bending folds
    env = np.clip(np.minimum.reduce([U - u0, u1 - U, V - v0, v1 - V]) / (0.25 * min(u1 - u0, v1 - v0)), 0, 1) ** 1.5
    shp = np.clip(0.55 + 0.45 * noise(U.shape, 30 * tpm, seed + 1), 0, 1.2)
    L["H"] = L["H"] + amp * env * shp * np.sin(2 * np.pi * (p + warp) / wave) * np.exp(-(np.abs(q - (q * env).mean()) / 400))


def f_pocket(L, U, V, tpm, rect, flap=20.0, puff=3.0, seed=0, **_):
    """flat sewn-on pocket with a flap: raised panel, binding edge stitch, flap shadow line"""
    d, s = seg_field(U, V, rrect(*rect, r=4)); inside = d < 0
    L["H"] = np.where(inside, L["H"] + puff * np.clip(-d / 6, 0, 1) ** 0.6, L["H"])
    f_stitch(L, U, V, tpm, offset_poly(rrect(*rect, r=4), 3.0), closed=True)
    u0, v0, u1, v1 = rect
    f_seam(L, U, V, tpm, [(u0 + 2, v1 - flap), (u1 - 2, v1 - flap)], depth=1.2, width=1.5)


FEATURES = dict(molle=f_molle, loop=f_loop, label=f_label, stitch=f_stitch, seam=f_seam, wrinkle=f_wrinkle, pocket=f_pocket)


# ------------------------------------------------------------------ one panel
def panel_layers(panel, tpm, seed=0):
    """height / id / thread / cover rasters of one panel at tpm texels per mm (row 0 = v max)"""
    P = np.asarray(panel["outline"], float)
    u0, v0 = P.min(0) - 2; u1, v1 = P.max(0) + 2
    W, H = int(np.ceil((u1 - u0) * tpm)), int(np.ceil((v1 - v0) * tpm))
    U, V = np.meshgrid(u0 + (np.arange(W) + 0.5) / tpm, v1 - (np.arange(H) + 0.5) / tpm)
    U = U.astype(np.float32); V = V.astype(np.float32)
    d, s = seg_field(U, V, P)
    L = dict(H=np.zeros(U.shape, np.float32), ID=np.full(U.shape, FABRIC, np.uint8), thread=np.zeros(U.shape, np.float32))
    edge = np.clip(-d / max(panel.get("puff_w", 12.0), 1), 0, 1)
    L["H"] += panel.get("puff_tex", 1.5) * np.sin(edge * np.pi / 2)               # pillow roll-off at the edges
    L["H"] += 0.25 * noise(U.shape, 20 * tpm / 4, seed)                           # fabric undulation
    b = panel.get("binding", 0.0)
    if b:
        inb = (d < 0) & (d > -b)
        L["H"] = np.where(inb, L["H"] + 0.8 * np.sqrt(np.clip(-d / 1.2, 0, 1)) * np.clip((b + d) / 1.0, 0, 1) + 0.5, L["H"])
        L["ID"][inb] = BINDING
        f_stitch(L, U, V, tpm, offset_poly(P, b - 2.2), closed=True)
    L["base"] = L["H"].copy()
    for i, f in enumerate(panel.get("features", ())):
        FEATURES[f["kind"]](L, U, V, tpm, seed=seed + 17 * i, **{k: v for k, v in f.items() if k != "kind"})
    L["ID"][L["thread"] > 0.5] = THREAD
    L["cover"] = d < 0.6 / tpm
    L["uv0"] = (float(u0), float(v0), float(u1), float(v1))
    del L["base"]
    return L


# ------------------------------------------------------------------ atlas -> PBR maps
def shelf_pack(sizes, W):
    """sizes [(w, h)] px -> positions in a W-wide atlas (tallest first), returns (pos, used height)"""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1]); pos = [None] * len(sizes)
    x = y = row_h = 0
    for i in order:
        w, h = sizes[i]
        if x + w > W:
            x, y, row_h = 0, y + row_h + 4, 0
        pos[i] = (x, y); x += w + 4; row_h = max(row_h, h)
    return pos, y + row_h


def fit_tpm(panels, size, lo=0.5, hi=12.0):
    """largest texel density at which every panel fits the size x size atlas"""
    for _ in range(24):
        mid = (lo + hi) / 2
        sz = [(int(np.ceil((np.ptp(np.asarray(p["outline"])[:, 0]) + 4) * mid)),
               int(np.ceil((np.ptp(np.asarray(p["outline"])[:, 1]) + 4) * mid))) for p in panels]
        ok = max(w for w, _ in sz) <= size and shelf_pack(sz, size)[1] <= size
        lo, hi = (mid, hi) if ok else (lo, mid)
    return lo


def compose(panels, size=2048, ss=2, seed=0, color_of=None):
    """all panels -> atlas maps (albedo sRGB 0..1, normal OpenGL 0..1, ORM, id) + per-panel uv rects (0..1, v up)"""
    from lab import tex as T
    tpm = fit_tpm(panels, size) * 0.98
    S = size * ss
    Hm = np.zeros((S, S), np.float32); ID = np.zeros((S, S), np.uint8); TH = np.zeros((S, S), np.float32)
    PID = np.full((S, S), -1, np.int32); COL = np.zeros((S, S, 3), np.float32)
    rects = []
    layers = [panel_layers(p, tpm * ss, seed + 101 * i) for i, p in enumerate(panels)]
    pos, _ = shelf_pack([(l["H"].shape[1], l["H"].shape[0]) for l in layers], S)
    for i, (l, (x, y), p) in enumerate(zip(layers, pos, panels)):
        h, w = l["H"].shape; c = l["cover"]
        sl = (slice(y, y + h), slice(x, x + w))
        Hm[sl][c] = l["H"][c]; ID[sl][c] = l["ID"][c]; TH[sl][c] = l["thread"][c]; PID[sl][c] = i
        COL[sl][c] = T.srgb_to_lin(np.array(T.hex2rgb(p["color"]), np.float32))
        u0, v0, u1, v1 = l["uv0"]
        rects.append(dict(name=p["name"], px=(x / ss, y / ss, w / ss, h / ss), mm=(u0, v0, u1, v1)))
    cover = PID >= 0
    idx = ndimage.distance_transform_edt(~cover, return_distances=False, return_indices=True)   # bleed (no seams)
    Hm, ID, TH, COL, PID = (a[idx[0], idx[1]] for a in (Hm, ID, TH, COL, PID))
    k = tpm * ss
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32) / k
    wv = np.zeros((S, S), np.float32)
    for mid, (_, alb, rough, pitch, amp) in MATS.items():
        if pitch:
            m = ID == mid
            if m.any():
                wv[m] = (weave(xx[m], yy[m], pitch, seed + mid) - 0.5) * amp
    Hfull = Hm + wv
    # normal: gradients only inside one panel
    def dd(axis, sgn):
        Hn = np.roll(Hfull, -sgn, axis); Pn = np.roll(PID, -sgn, axis); return np.where(Pn == PID, Hn, Hfull)
    dx = (dd(1, 1) - dd(1, -1)) * 0.5 * k; dy = (dd(0, -1) - dd(0, 1)) * 0.5 * k
    nrm = np.stack([-dx, -dy, np.ones_like(Hfull)], -1); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    # cavity AO from height, dirt, fray
    cav = np.clip((ndimage.gaussian_filter(Hm, 3 * k) - Hm) / 0.8, 0, 1)
    ao = np.clip(1 - 0.55 * cav - 0.25 * TH * 0, 0.25, 1)
    dirt = np.clip(noise((S, S), 60 * k, seed + 5) * 0.15 + 0.15 + 0.5 * cav, 0, 1)
    albx = np.choose(ID, [MATS[i][1] for i in range(len(MATS))]).astype(np.float32)
    rough = np.choose(ID, [MATS[i][2] for i in range(len(MATS))]).astype(np.float32)
    var = 1 + 0.06 * noise((S, S), 8 * k, seed + 9) + 0.25 * wv
    alb = COL * (albx * var)[..., None] * (1 - 0.18 * dirt)[..., None] * (0.75 + 0.25 * ao)[..., None]
    rough = np.clip(rough + 0.05 * dirt - 0.04 * wv, 0, 1)
    down = lambda a: a.reshape(size, ss, size, ss, *a.shape[2:]).mean((1, 3)) if ss > 1 else a
    nrm = down(nrm); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    out = dict(albedo=T.lin_to_srgb(np.clip(down(alb), 0, 1)), normal=nrm * 0.5 + 0.5,
               orm=np.stack([down(ao), down(rough), np.zeros((size, size), np.float32)], -1), height=down(Hfull),
               id=ID[::ss, ::ss])
    return out, rects, tpm
