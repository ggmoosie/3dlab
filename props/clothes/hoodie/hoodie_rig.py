"""Skinning: heat weights on the proxy body (weighting rig), transferred to the hoodie, then a cleanup pass. QC posing."""
import math
import numpy as np
import bpy
from mathutils import Vector, Matrix, Quaternion
from mathutils.bvhtree import BVHTree
import hoodie_body as B

# bones the hoodie may use: core + twist (no fingers, no legs below the thigh)
ALLOWED = ["pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head"] + \
    [b.format(s=s) for s in "lr" for b in ("clavicle_{s}", "upperarm_{s}", "upperarm_twist_01_{s}", "upperarm_twist_02_{s}",
                                           "lowerarm_{s}", "lowerarm_twist_01_{s}", "lowerarm_twist_02_{s}", "hand_{s}", "thigh_{s}")]
CAP = {"thigh_l": 0.45, "thigh_r": 0.45, "hand_l": 0.18, "hand_r": 0.18, "head": 0.12}


def bone_heat(pV, pT, names, sdf):
    """Automatic (bone-heat) weights, Baran & Popovic 2007 - what Blender's 'automatic weights' does: each vertex is
    attached to its nearest *visible* bone segment with strength 1/d^2, then the attachment diffuses over the surface:
    (L + H) w_j = H p_j  (L = graph Laplacian of the proxy, H = diag heat). Visibility = the straight line from the
    vertex to the segment stays inside the body (sdf < 0.3 at 3 samples)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    n, nb = len(pV), len(names)
    heads = np.array([B.J[b] for b in names]); tails = np.array([B.seg_tail(b) for b in names])
    D = np.full((n, nb), np.inf)
    for j in range(nb):
        a, b = heads[j], tails[j]; ab = b - a
        t = np.clip(((pV - a) @ ab) / (ab @ ab), 0, 1)
        q = a + t[:, None] * ab
        d = np.linalg.norm(pV - q, axis=1)
        vis = np.ones(n, bool)
        for f in (0.3, 0.55, 0.8):
            vis &= sdf(pV + (q - pV) * f) < 0.3
        D[:, j] = np.where(vis, d, np.inf)
    nearest = D.min(1)
    bad = ~np.isfinite(nearest)                                  # nothing visible: fall back to plain distance
    if bad.any():
        for j in range(nb):
            a, b = heads[j], tails[j]; ab = b - a
            t = np.clip(((pV[bad] - a) @ ab) / (ab @ ab), 0, 1)
            D[bad, j] = np.linalg.norm(pV[bad] - (a + t[:, None] * ab), axis=1)
        nearest = D.min(1)
    Pm = (D <= nearest[:, None] * 1.0001).astype(float)
    Pm /= Pm.sum(1, keepdims=True)
    H = 1.0 / np.maximum(nearest, 0.5) ** 2
    e = np.concatenate([pT[:, [0, 1]], pT[:, [1, 2]], pT[:, [2, 0]]])
    e = np.unique(np.sort(e, 1), axis=0)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)); A = (A + A.T).tocsr()
    deg = np.asarray(A.sum(1)).ravel()
    # uniform Laplacian scaled by the mean edge length^2 so heat and diffusion are in the same units (1/cm^2)
    el = np.linalg.norm(pV[e[:, 0]] - pV[e[:, 1]], axis=1).mean()
    L = (sp.diags(deg) - A) / (el * el)
    M = (L + sp.diags(H)).tocsc()
    lu = splu(M)
    W = np.stack([lu.solve(H * Pm[:, j]) for j in range(nb)], 1)
    W = np.clip(W, 0, None)
    W[W < 1e-3] = 0
    return W / np.maximum(W.sum(1, keepdims=True), 1e-9)


def heat(proxy, wrig):
    bpy.ops.object.select_all(action="DESELECT")
    proxy.select_set(True); wrig.select_set(True)
    bpy.context.view_layer.objects.active = wrig
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    return len(proxy.vertex_groups)


def group_matrix(o, names):
    """(nverts, nbones) weights of o's vertex groups in `names` order."""
    idx = {g.index: names.index(g.name) for g in o.vertex_groups if g.name in names}
    W = np.zeros((len(o.data.vertices), len(names)))
    for v in o.data.vertices:
        for g in v.groups:
            if g.group in idx:
                W[v.index, idx[g.group]] = g.weight
    return W


def tri_mesh(V, F):
    T = []
    for f in F:
        for k in range(1, len(f) - 1):
            T.append((f[0], f[k], f[k + 1]))
    return np.asarray(T)


def nearest_interp(src_V, src_T, src_W, P):
    """Barycentric-interpolated src_W at the nearest surface point of each P."""
    bvh = BVHTree.FromPolygons([Vector(v) for v in src_V], [list(map(int, t)) for t in src_T])
    out = np.zeros((len(P), src_W.shape[1]))
    for i, p in enumerate(P):
        q, _, fi, _ = bvh.find_nearest(Vector(p))
        a, b, c = src_V[src_T[fi]]
        v0, v1, v2 = b - a, c - a, np.array(q) - a
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01 or 1e-12
        wb = (d11 * d20 - d01 * d21) / den; wc = (d00 * d21 - d01 * d20) / den
        wa = 1 - wb - wc
        w = np.clip([wa, wb, wc], 0, 1); w /= w.sum()
        out[i] = w @ src_W[src_T[fi]]
    return out


def cleanup(W, edges=None, smooth=2, names=ALLOWED, limit=4):
    W = W.copy()
    for n, c in CAP.items():
        j = names.index(n)
        over = W[:, j] > c
        W[over, j] = c
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    if edges is not None and smooth:
        deg = np.bincount(edges.ravel(), minlength=len(W)).astype(float)[:, None]
        for _ in range(smooth):
            acc = np.zeros_like(W); np.add.at(acc, edges[:, 0], W[edges[:, 1]]); np.add.at(acc, edges[:, 1], W[edges[:, 0]])
            W = np.where(deg > 0, 0.5 * W + 0.5 * acc / np.maximum(deg, 1), W)
    # limit influences, prune crumbs, normalise
    order = np.argsort(-W, 1)
    keep = np.zeros_like(W, bool)
    np.put_along_axis(keep, order[:, :limit], True, 1)
    W = np.where(keep & (W >= 0.01), W, 0.0)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    return W


def write_groups(o, W, names=ALLOWED):
    o.vertex_groups.clear()
    used = [j for j in range(len(names)) if W[:, j].max() > 0]
    for j in used:
        g = o.vertex_groups.new(name=names[j])
        nz = np.nonzero(W[:, j])[0]
        for i in nz:
            g.add([int(i)], float(W[i, j]), "REPLACE")
    return [names[j] for j in used]


def bind(o, rig):
    o.parent = rig
    m = o.modifiers.new("Armature", "ARMATURE")
    m.object = rig
    return m


# ------------------------------------------------------------------------------------------------------------ posing
def pose(rig, rots, moves=None):
    """rots: {bone: (world axis, degrees)} applied parent-first about each bone's (posed) head. moves: {bone: world offset}."""
    pb = rig.pose.bones
    for b in pb:
        b.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    rest = {b.name: b.bone.matrix_local.copy() for b in pb}
    delta = {}
    order = []

    def walk(b):
        order.append(b)
        for c in b.children:
            walk(c)
    for b in pb:
        if b.parent is None:
            walk(b)
    for b in order:
        D = delta[b.parent.name] if b.parent else Matrix.Identity(4)
        M = D @ rest[b.name]
        if moves and b.name in moves:
            M = Matrix.Translation(Vector(moves[b.name])) @ M
        if b.name in rots:
            ax, deg = rots[b.name]
            h = M.to_translation()
            R = Matrix.Translation(h) @ Quaternion(Vector(ax).normalized(), math.radians(deg)).to_matrix().to_4x4() @ Matrix.Translation(-h)
            M = R @ M
        delta[b.name] = M @ rest[b.name].inverted()
        b.matrix = M
        bpy.context.view_layer.update()


def _dir(a, b):
    v = B.J[b] - B.J[a]
    return v / np.linalg.norm(v)


def poses():
    """The QC deformation set (world axes, Blender frame: +X = left, -Y = front, +Z = up)."""
    ud = _dir("upperarm_l", "lowerarm_l")
    down = math.degrees(math.atan2(-ud[2], math.hypot(ud[0], ud[1])))       # A-pose arm angle below horizontal
    P = {"rest": ({}, None)}
    P["arms_up_90"] = ({"clavicle_l": ((0, 1, 0), -12), "upperarm_l": ((0, 1, 0), -(down - 12)),
                        "clavicle_r": ((0, 1, 0), 12), "upperarm_r": ((0, 1, 0), (down - 12))}, None)
    rf = {}
    for s, sg in (("l", 1), ("r", -1)):
        a = Vector(_dir("upperarm_" + s, "lowerarm_" + s))
        q = a.rotation_difference(Vector((0.12 * sg, -1.0, -0.05)).normalized())
        ax, ang = q.to_axis_angle()
        rf["clavicle_" + s] = ((0, 0, 1), 8 * sg)
        rf["upperarm_" + s] = (tuple(ax), math.degrees(ang) - 4)
    P["arms_forward"] = (rf, None)
    cr = {"thigh_l": ((1, 0, 0), -55), "thigh_r": ((1, 0, 0), -55), "calf_l": ((1, 0, 0), 60), "calf_r": ((1, 0, 0), 60),
          "foot_l": ((1, 0, 0), -5), "foot_r": ((1, 0, 0), -5),
          "spine_01": ((0, 0, 1), 6), "spine_02": ((0, 0, 1), 6), "spine_03": ((0, 0, 1), 6), "spine_04": ((0, 0, 1), 6),
          "spine_05": ((0, 0, 1), 6)}
    cr["spine_02"] = ((0.0, 0, 1), 6)
    cr["spine_03"] = ((1, 0, 0.35), 12)
    P["crouch_twist"] = (cr, {"pelvis": (0, 0, -16)})
    el = {}
    for s in "lr":
        ud = Vector(_dir("upperarm_" + s, "lowerarm_" + s)); fd = Vector(_dir("lowerarm_" + s, "hand_" + s))
        ax = ud.cross(fd).normalized()
        el["lowerarm_" + s] = (tuple(ax), 120 - math.degrees(ud.angle(fd)))
    P["elbow_120"] = (el, None)
    return P
