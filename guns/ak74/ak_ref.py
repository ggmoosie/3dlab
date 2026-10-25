"""ak_ref.py - the AK-74M profile master and the parametric-outline machinery.

Photos (refs_user/ak74/, gitignored, never committed): the right profile 163254 is the master (near-orthographic, muzzle to
the right, so we see the gun's right side). Calibration: AK-74M overall length 943 mm (brake face .. buttplate face) =
subpixel columns 1179.4 .. 199.6 -> S = 0.9624 mm/px. Bore row 286.1 (barrel centre between gas block and front sight,
level to 0.3 px over 50 px). B frame (mm): x gun-LEFT, y toward the muzzle, z up; bore axis z = 0; y = 0 at the receiver
rear face (photo column 460).  Photo x -> y = (x - X0) * S;  row -> z = (ROW_BORE - row) * S.

Outline: a part's side profile is NOT a traced pixel blob. It is a clean polygon of straight edges and circular arcs
(per-edge sagitta) with per-corner fillet radii, vertices read off the pixel grids (pxgrid.py). Edges that lie on the
photo silhouette are flagged 'sil'; fit() moves only the vertices of silhouette edges so the straight lines / arcs sit on
the photo edge (least squares on the photo's subpixel signed distance, edge ends excluded), internal edges stay as
measured. field() rasterises the filleted outline to a Field2D signed distance in (y, z) mm."""
import os, math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize
HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S

REFD = os.path.join(HERE, "refs_user", "ak74")
PHOTO = {k: os.path.join(REFD, f"Screenshot 2026-10-25 {v}.png") for k, v in
         dict(R="163254", L="163245", H="163300", F="163304", GRIP="163309", RS="163315", DC="163320", RSL="163326",
              SEL="163334", GAS="163337", MUZ="163339").items()}
PHOTO["ANAT"] = os.path.join(REFD, "akt4anatomy.jpg")

OAL = 943.0
X_BUTT, X_MUZ = 199.6, 1179.4
S_MM = OAL / (X_MUZ - X_BUTT)
X0 = 460.0                     # receiver rear face -> y 0
# the master is rotated 0.554 deg (muzzle down): barrel (2 segments) + brake centres lie on one line, residual 0.10 px,
# row = 285.98 + 0.00967 (x - 1030). In that frame the dust-cover top is parallel to the bore, as on the rifle.
BORE_SLOPE = 0.00967
ROW_BORE = 285.98 + BORE_SLOPE * (X0 - 1030.0)          # bore row at x = X0 (280.47)
_CS = 1.0 / np.hypot(1.0, BORE_SLOPE)


def px2b(x, row):
    u = np.asarray(x, float) - X0; v = ROW_BORE - np.asarray(row, float)
    return (u - v * BORE_SLOPE) * _CS * S_MM, (u * BORE_SLOPE + v) * _CS * S_MM


def b2px(y, z):
    a = np.asarray(y, float) / S_MM / _CS; c = np.asarray(z, float) / S_MM / _CS
    # solve u - s v = a, s u + v = c
    u = (a + BORE_SLOPE * c) / (1 + BORE_SLOPE ** 2); v = c - BORE_SLOPE * u
    return u + X0, ROW_BORE - v


MUZZLE_Y = float(px2b(X_MUZ, 285.98 + BORE_SLOPE * (X_MUZ - 1030.0))[0])
BUTT_Y = float(px2b(X_BUTT, 336.0)[0])


def bp(x, row):
    y, z = px2b(x, row)
    return float(y), float(z)


# ------------------------------------------------------------------ photo masks
_IMG, _COV, _MASK, _SDF = {}, {}, {}, {}
# Photo-mask repairs (the photo is the truth, but a highlight is not a hole):
#  FILL_SEEDS  - bright interior regions larger than the auto-fill size (polished latch plate, pins) -> solid
#  SOLID_BOXES - (x0, r0, x1, r1): every column is solid between its first and last foreground row (specular streaks on
#                the dust cover and receiver that touch the outline)
#  LIGHT_BOXES - (x0, r0, x1, r1): the silver/light trigger guard and mag-catch tip read at coverage > 0.12
#  holes <= AUTO_FILL px are specks (pins, rivet highlights) and are filled; larger holes are real see-through openings
FILL_SEEDS = {"R": [(470, 290), (477, 293)], "L": []}
SOLID_BOXES = {"R": [(462, 238, 700, 300), (780, 241, 893, 270)], "L": [(620, 230, 880, 300), (450, 230, 560, 268)]}
LIGHT_BOXES = {"R": [(528, 335, 600, 358), (605, 338, 620, 360)], "L": [(758, 325, 822, 348), (748, 330, 760, 352)]}
AUTO_FILL = 40


def image(k):
    if k not in _IMG:
        _IMG[k] = np.asarray(Image.open(PHOTO[k]).convert("RGB")).astype(np.float32)
    return _IMG[k]


def coverage(k):
    """soft foreground coverage 0..1 from the luminance against the white sweep (antialiased edges are fractional)"""
    if k not in _COV:
        a = image(k); L = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        _COV[k] = np.clip((250.0 - L) / (250.0 - 75.0), 0, 1)
    return _COV[k]


def mask(k):
    """photo silhouette (coverage > 0.5, the largest component), bright interior details filled back in"""
    if k in _MASK:
        return _MASK[k]
    c = coverage(k)
    m = c > 0.5
    for x0, r0, x1, r1 in LIGHT_BOXES.get(k, []):
        m[r0:r1, x0:x1] |= c[r0:r1, x0:x1] > 0.12
    lab, n = ndimage.label(m)
    size = ndimage.sum(m, lab, range(1, n + 1))
    m = lab == (1 + int(np.argmax(size)))
    for x0, r0, x1, r1 in SOLID_BOXES.get(k, []):
        sub = m[r0:r1, x0:x1]
        has = sub.any(0)
        top = np.argmax(sub, 0); bot = sub.shape[0] - 1 - np.argmax(sub[::-1], 0)
        rr = np.arange(sub.shape[0])[:, None]
        m[r0:r1, x0:x1] |= has[None] & (rr >= top[None]) & (rr <= bot[None])
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hn = ndimage.label(holes)
    hs = ndimage.sum(holes, hl, range(1, hn + 1))
    m |= np.isin(hl, 1 + np.nonzero(hs <= AUTO_FILL)[0])
    for x, r in FILL_SEEDS.get(k, []):
        if hl[r, x]:
            m |= hl == hl[r, x]
    _MASK[k] = m
    return m


def photo_sdf(k, up=4):
    """subpixel signed distance (px, < 0 inside) of the photo silhouette: upsampled coverage, 0.5 iso-line"""
    if k in _SDF:
        return _SDF[k]
    m = mask(k)
    c = np.maximum(coverage(k), ndimage.binary_erosion(m, iterations=1)) * mask_dilated(k)
    H, W = c.shape
    big = np.asarray(Image.fromarray((c * 255).astype(np.uint8)).resize((W * up, H * up), Image.BICUBIC)) > 127
    d = (ndimage.distance_transform_edt(~big) - ndimage.distance_transform_edt(big)) / up
    _SDF[k] = (d.astype(np.float32), up)
    return _SDF[k]


def mask_dilated(k):
    return ndimage.binary_dilation(mask(k), iterations=2).astype(np.float32)


def sample_sdf(k, xs, rows):
    """bilinear sample of photo_sdf at photo px (x right, row down); pixel centres at integer + 0.5"""
    d, up = photo_sdf(k)
    fx = (np.asarray(xs, float) + 0.5) * up - 0.5; fy = (np.asarray(rows, float) + 0.5) * up - 0.5
    ix = np.clip(np.floor(fx).astype(int), 0, d.shape[1] - 2); iy = np.clip(np.floor(fy).astype(int), 0, d.shape[0] - 2)
    tx = np.clip(fx - ix, 0, 1); ty = np.clip(fy - iy, 0, 1)
    return (d[iy, ix] * (1 - tx) * (1 - ty) + d[iy, ix + 1] * tx * (1 - ty) + d[iy + 1, ix] * (1 - tx) * ty + d[iy + 1, ix + 1] * tx * ty)


# ------------------------------------------------------------------ parametric outlines
def _arc_pts(p0, p1, sag, n):
    """points from p0 to p1 (excluded) along a circular arc of sagitta sag (px; > 0 bulges to the left of p0->p1)"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n, endpoint=False)
    if abs(sag) < 1e-6:
        return p0 + (p1 - p0) * t[:, None]
    ch = p1 - p0; L = np.linalg.norm(ch); nrm = np.array([ch[1], -ch[0]]) / L       # left normal in photo coords (row down)
    R = (sag * sag + (L / 2) ** 2) / (2 * sag)
    c = (p0 + p1) / 2 + nrm * (sag - R)
    a0 = math.atan2(*(p0 - c)[::-1]); a1 = math.atan2(*(p1 - c)[::-1])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    if abs(da) < 1e-9:
        da = 2 * math.pi
    # choose the sweep that passes through the sagitta point
    mid = (p0 + p1) / 2 + nrm * sag
    am = math.atan2(*(mid - c)[::-1])
    def on(a0, da, am):
        x = (am - a0) % (2 * math.pi) if da > 0 else (a0 - am) % (2 * math.pi)
        return x <= abs(da)
    if not on(a0, da, am):
        da = da - 2 * math.pi if da > 0 else da + 2 * math.pi
    a = a0 + da * t
    return c + abs(R) * np.stack([np.cos(a), np.sin(a)], 1)


class Outline:
    """closed outline in photo-R px. pts [(x, row)], r per-vertex fillet (mm), sil per-edge flag, sag per-edge arc sagitta (px).
    Edge i runs pts[i] -> pts[i+1]."""

    def __init__(self, pts, r=0.0, sil=True, sag=0.0, name="", fix=()):
        self.p = np.array(pts, float); n = len(self.p)
        self.r = np.broadcast_to(np.asarray(r, float), (n,)).copy()
        self.sil = np.broadcast_to(np.asarray(sil, bool), (n,)).copy()
        self.sag = np.broadcast_to(np.asarray(sag, float), (n,)).copy()
        self.fix = set(fix)
        self.name = name
        self.fit_rms = None

    def edge_pts(self, i, n=16, t0=0.0, t1=1.0):
        a, b = self.p[i], self.p[(i + 1) % len(self.p)]
        pts = _arc_pts(a, b, self.sag[i], 64)
        pts = np.vstack([pts, b])
        k = np.linspace(t0, t1, n) * (len(pts) - 1)
        i0 = np.clip(np.floor(k).astype(int), 0, len(pts) - 2); f = (k - i0)[:, None]
        return pts[i0] * (1 - f) + pts[i0 + 1] * f

    def raw_poly(self, n=24):
        out = []
        for i in range(len(self.p)):
            out.append(_arc_pts(self.p[i], self.p[(i + 1) % len(self.p)], self.sag[i], n if abs(self.sag[i]) > 1e-6 else 1))
        return np.vstack(out)

    def poly_mm(self, n=24):
        """filleted polygon in (y, z) mm.  Fillets: at each vertex with r > 0 the corner is replaced by a circular arc
        tangent to both neighbouring edges (their local directions at the vertex)"""
        nv = len(self.p)
        segs = []
        for i in range(nv):
            s = np.vstack([_arc_pts(self.p[i], self.p[(i + 1) % nv], self.sag[i], n if abs(self.sag[i]) > 1e-6 else 1),
                           self.p[(i + 1) % nv]])
            segs.append(np.stack(px2b(s[:, 0], s[:, 1]), 1))
        tl = np.zeros(nv); arcs = [None] * nv
        for i in range(nv):
            r = self.r[i]
            if r <= 0:
                continue
            s, prev = segs[i], segs[i - 1]
            v = s[0]; a = prev[-2] - v; b = s[1] - v
            la, lb = np.linalg.norm(prev[0] - v), np.linalg.norm(s[-1] - v)
            a /= np.linalg.norm(a); b /= np.linalg.norm(b)
            ang = math.acos(np.clip(np.dot(a, b), -1, 1))
            if not (1e-3 < ang < math.pi - 1e-3):
                continue
            t = min(r / math.tan(ang / 2), 0.45 * la, 0.45 * lb)
            rr = t * math.tan(ang / 2)
            pa, pb = v + a * t, v + b * t
            bis = (a + b) / np.linalg.norm(a + b)
            c = v + bis * (rr / math.sin(ang / 2))
            a0 = math.atan2(*(pa - c)[::-1]); a1 = math.atan2(*(pb - c)[::-1])
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            k = max(4, int(abs(da) / 0.1))
            u = np.linspace(0, 1, k)
            arcs[i] = c + rr * np.stack([np.cos(a0 + da * u), np.sin(a0 + da * u)], 1)
            tl[i] = t
        P = []
        for i in range(nv):
            if arcs[i] is not None:
                P.append(arcs[i])
            else:
                P.append(segs[i][:1])
            s = segs[i]; j = (i + 1) % nv
            if len(s) > 2:
                inner = s[1:-1]
                keep = (np.linalg.norm(inner - s[0], axis=1) > tl[i]) & (np.linalg.norm(inner - s[-1], axis=1) > tl[j])
                P.append(inner[keep])
        return np.vstack([p for p in P if len(p)])

    def field(self, res=0.1, pad=6.0, soften=0.0):
        P = self.poly_mm()
        y0, z0 = P.min(0) - pad; y1, z1 = P.max(0) + pad
        W, H = int(math.ceil((y1 - y0) / res)), int(math.ceil((z1 - z0) / res))
        im = Image.new("L", (W, H), 0)
        ImageDraw.Draw(im).polygon([((y - y0) / res, (z1 - z) / res) for y, z in P], fill=255)
        m = np.asarray(im) > 127
        if soften > 0:
            m = ndimage.gaussian_filter(m.astype(np.float32), soften / res) > 0.5
        d = (ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)) * res
        return S.Field2D(d.astype(np.float32), y0, z1, res, outside=50.0)

    def fit(self, k="R", n=16, reg=0.02, verbose=True):
        """move the vertices of silhouette edges (and the sagitta of silhouette arcs) so the edges sit on the photo
        silhouette (soft-L1 least squares on the photo's subpixel signed distance; 12 % of each edge end excluded)"""
        nv = len(self.p)
        free = np.zeros(nv, bool)
        for i in range(nv):
            if self.sil[i]:
                free[i] = free[(i + 1) % nv] = True
        for i in self.fix:
            free[i] = False
        idx = np.nonzero(free)[0]
        sidx = np.nonzero(self.sil & (np.abs(self.sag) > 1e-6))[0]
        if not len(idx) and not len(sidx):
            return self
        p0, s0 = self.p.copy(), self.sag.copy()
        nreg = 2 * len(idx) + len(sidx)

        def res(v):
            self.p[idx] = v[:2 * len(idx)].reshape(-1, 2)
            self.sag[sidx] = v[2 * len(idx):]
            out = []
            for i in range(nv):
                if not self.sil[i]:
                    continue
                q = self.edge_pts(i, n, 0.12, 0.88)
                out.append(sample_sdf(k, q[:, 0], q[:, 1]))
            out.append(reg * (self.p[idx] - p0[idx]).ravel())
            out.append(reg * (self.sag[sidx] - s0[sidx]))
            return np.concatenate(out)
        x0 = np.concatenate([p0[idx].ravel(), s0[sidx]])
        r0 = np.sqrt(np.mean(res(x0)[:-nreg] ** 2)) if nreg else 0
        sol = optimize.least_squares(res, x0, loss="soft_l1", f_scale=1.0, max_nfev=600)
        r = res(sol.x)[:-nreg] if nreg else res(sol.x)
        self.fit_rms = float(np.sqrt(np.mean(r ** 2)))
        if verbose:
            mv = np.abs(self.p - p0).max()
            print(f"[fit] {self.name:12s} edge rms {r0:.2f} -> {self.fit_rms:.2f} px  max vertex move {mv:.1f} px")
        return self

    def edge_report(self, k="R", n=16):
        """per silhouette edge: mean signed photo distance (px; + = edge outside the photo silhouette)"""
        out = []
        for i in range(len(self.p)):
            if self.sil[i]:
                q = self.edge_pts(i, n, 0.12, 0.88)
                d = sample_sdf(k, q[:, 0], q[:, 1]); out.append((i, float(d.mean()), float(np.abs(d).max())))
        return out

    def as_list(self):
        return [(round(float(x), 2), round(float(y), 2)) for x, y in self.p]


def fitted(name, ol, k="R"):
    """fit once, cache the fitted vertices (work/outlines.json) so parts.py is fast and deterministic"""
    import json
    path = os.path.join(HERE, "work", "outlines.json")
    db = json.load(open(path)) if os.path.exists(path) else {}
    key = name
    sig = [ol.p.round(2).tolist(), ol.sil.tolist(), ol.sag.round(2).tolist(), sorted(ol.fix)]
    if key in db and db[key]["sig"] == sig:
        ol.p = np.array(db[key]["p"], float); ol.fit_rms = db[key]["rms"]
        ol.sag = np.array(db[key].get("sag", ol.sag), float)
        return ol
    ol.name = name
    ol.fit(k)
    db[key] = dict(sig=sig, p=ol.p.tolist(), sag=ol.sag.tolist(), rms=ol.fit_rms)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(db, open(path, "w"))
    return ol


# ------------------------------------------------------------------ left profile (163245) -> B frame
# mirrored photo (muzzle on the left), rotated 0.37 deg (muzzle down: brake 277.6 @ x 200, barrel 277.25 @ 300, 276.25 @ 410).
# Landmarks: brake face x 154.3 <-> MUZZLE_Y, receiver rear face x 886.5 <-> y -1.6; dust-cover top 233.5 <-> z 42.25.
# Horizontal 0.9499 mm/px, vertical 0.9668 mm/px (the left photo is ~1.7 % wider than the master: display scaling; the
# compare stage fits a full homography + radial term per photo)
L_X_MUZ, L_X_REAR = 154.3, 886.5
L_SLOPE = -0.0064
L_ROW_MUZ = 277.6 + L_SLOPE * (L_X_MUZ - 200.0)
L_SH = (MUZZLE_Y + 1.6) / (L_X_REAR - L_X_MUZ)
L_SV = L_SH
_LMAP = None


def lp2b(x, row):
    """left-photo px -> B (y, z). Uses the homography + radial fit stored by compare.py (work/map_L.json) when present
    (the left photo has perspective: the muzzle end is ~5 % larger than the stock end), else the similarity above."""
    global _LMAP
    if _LMAP is None:
        import json
        p = os.path.join(HERE, "work", "map_L.json")
        _LMAP = json.load(open(p)) if os.path.exists(p) else {}
    if _LMAP:
        Hm = np.array(_LMAP["photo_to_b"]).reshape(3, 3)
        cx, cy, k1 = _LMAP["cx"], _LMAP["cy"], _LMAP["k1"]
        dx, dy = x - cx, row - cy; f = 1 + k1 * (dx * dx + dy * dy)
        q = Hm @ np.array([cx + dx * f, cy + dy * f, 1.0])
        return float(q[0] / q[2]), float(q[1] / q[2])
    u = x - L_X_MUZ; v = (L_ROW_MUZ + L_SLOPE * u) - row
    return float(MUZZLE_Y - u * L_SH), float(v * L_SV)
