"""Blender mesh from the numpy build, UV layout (seams = sewing seams + cuts), and a numpy UV rasteriser for texture work."""
import math
import numpy as np
import bpy
import hoodie_geo as G

# texel density relative to the outer fabric (inside of the body is rarely seen; the hood lining is)
DENSITY = {G.R_INNER: 0.25, G.R_POCK_IN: 0.25, G.R_LINING: 0.65, G.R_STRING: 1.3, G.R_KNOT: 1.3, G.R_AGLET: 1.5, G.R_GROM: 1.8}


def make_mesh(name, M, coll=None, uv=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(M["V"].tolist(), [], [list(f) for f in M["F"]])
    me.validate(clean_customdata=False)
    seams = M["seams"]
    for e in me.edges:
        a, b = e.vertices
        if (min(a, b), max(a, b)) in seams:
            e.use_seam = True
    at = me.attributes.new("reg", "INT", "FACE")
    at.data.foreach_set("value", np.asarray(M["reg"], np.int32))
    me.polygons.foreach_set("use_smooth", np.ones(len(me.polygons), bool))
    if uv:
        me.uv_layers.new(name="UVMap")
    o = bpy.data.objects.new(name, me)
    (coll or bpy.context.scene.collection).objects.link(o)
    return o


def islands(nf, F, seams):
    """Face islands: faces joined across edges that are not seams (union-find)."""
    par = list(range(nf))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    owner = {}
    for fi, f in enumerate(F):
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            e = (min(a, b), max(a, b))
            if e in seams:
                continue
            if e in owner:
                ra, rb = find(owner[e]), find(fi)
                if ra != rb:
                    par[ra] = rb
            else:
                owner[e] = fi
    roots = np.array([find(i) for i in range(nf)])
    _, isl = np.unique(roots, return_inverse=True)
    return isl


def layout(o, M, margin=0.0035):
    """Angle-based unwrap with the sewing seams, islands scaled to one texel density (x DENSITY by region), packed."""
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.001)
    bpy.ops.object.mode_set(mode="OBJECT")
    me = o.data
    nl = len(me.loops)
    uv = np.zeros(nl * 2); me.uv_layers["UVMap"].data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    ls = np.zeros(len(me.polygons), int); me.polygons.foreach_get("loop_start", ls)
    lt = np.zeros(len(me.polygons), int); me.polygons.foreach_get("loop_total", lt)
    lv = np.zeros(nl, int); me.loops.foreach_get("vertex_index", lv)
    V = M["V"]
    isl = islands(len(me.polygons), M["F"], M["seams"])
    reg = np.asarray(M["reg"])
    a3 = np.zeros(len(me.polygons)); auv = np.zeros(len(me.polygons))
    for fi in range(len(me.polygons)):
        idx = np.arange(ls[fi], ls[fi] + lt[fi])
        p = V[lv[idx]]; q = uv[idx]
        for k in range(1, lt[fi] - 1):
            a3[fi] += np.linalg.norm(np.cross(p[k] - p[0], p[k + 1] - p[0])) / 2
            auv[fi] += abs((q[k, 0] - q[0, 0]) * (q[k + 1, 1] - q[0, 1]) - (q[k + 1, 0] - q[0, 0]) * (q[k, 1] - q[0, 1])) / 2
    for i in range(isl.max() + 1):
        fs = np.where(isl == i)[0]
        r = np.bincount(reg[fs]).argmax()
        k = DENSITY.get(int(r), 1.0) * math.sqrt(a3[fs].sum() / max(auv[fs].sum(), 1e-12))
        loops = np.concatenate([np.arange(ls[f], ls[f] + lt[f]) for f in fs])
        c = uv[loops].mean(0)
        uv[loops] = (uv[loops] - c) * k
    me.uv_layers["UVMap"].data.foreach_set("uv", uv.ravel())
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, scale=True, margin=margin, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = np.zeros(nl * 2); me.uv_layers["UVMap"].data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    # texel density of the outer fabric (UV units per cm)
    out = np.isin(reg, [G.R_TORSO, G.R_SLEEVE, G.R_HOOD])
    auv2 = np.zeros(len(me.polygons))
    for fi in np.where(out)[0]:
        q = uv[ls[fi]:ls[fi] + lt[fi]]
        for k in range(1, lt[fi] - 1):
            auv2[fi] += abs((q[k, 0] - q[0, 0]) * (q[k + 1, 1] - q[0, 1]) - (q[k + 1, 0] - q[0, 0]) * (q[k, 1] - q[0, 1])) / 2
    dens = math.sqrt(auv2[out].sum() / a3[out].sum())
    return dict(uv=uv, ls=ls, lt=lt, lv=lv, isl=isl, uv_per_cm=dens)


def tris_of(L, M, attrs):
    """Fan-triangulate polygons -> (T,3) loop indices, face ids."""
    T, fid = [], []
    for fi in range(len(L["ls"])):
        s, n = L["ls"][fi], L["lt"][fi]
        for k in range(1, n - 1):
            T.append((s, s + k, s + k + 1)); fid.append(fi)
    return np.array(T), np.array(fid)


def read_layout(o):
    me = o.data
    nl = len(me.loops)
    uv = np.zeros(nl * 2); me.uv_layers["UVMap"].data.foreach_get("uv", uv)
    ls = np.zeros(len(me.polygons), int); me.polygons.foreach_get("loop_start", ls)
    lt = np.zeros(len(me.polygons), int); me.polygons.foreach_get("loop_total", lt)
    lv = np.zeros(nl, int); me.loops.foreach_get("vertex_index", lv)
    reg = np.zeros(len(me.polygons), np.int32); me.attributes["reg"].data.foreach_get("value", reg)
    return dict(uv=uv.reshape(-1, 2), ls=ls, lt=lt, lv=lv, reg=reg)


def raster(L, lattr, size):
    """Rasterise per-loop attributes (nloops, k) into UV space. Returns img (size,size,k), face id (-1 = empty).
    Row 0 = v 1 (image convention)."""
    T, fid = tris_of(L, None, None)
    uvp = L["uv"][T] * size                                 # (T,3,2) pixel coords, v up
    uvp[..., 1] = size - uvp[..., 1]
    A = lattr[T]                                            # (T,3,k)
    k = lattr.shape[1]
    img = np.zeros((size, size, k), np.float32)
    fimg = np.full((size, size), -1, np.int32)
    for t in range(len(T)):
        p = uvp[t]
        x0, y0 = np.floor(p.min(0)).astype(int); x1, y1 = np.ceil(p.max(0)).astype(int)
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, size), min(y1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = p
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / d
        l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / d
        l3 = 1 - l1 - l2
        m = (l1 >= -1e-4) & (l2 >= -1e-4) & (l3 >= -1e-4)
        if not m.any():
            continue
        val = l1[..., None] * A[t, 0] + l2[..., None] * A[t, 1] + l3[..., None] * A[t, 2]
        sub = img[y0:y1, x0:x1]; sub[m] = val[m]
        fs = fimg[y0:y1, x0:x1]; fs[m] = fid[t]
    return img, fimg


def dilate(img, valid, px=None):
    """Fill empty texels with the nearest valid texel (padding for mips / bilinear)."""
    from scipy import ndimage
    _, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
    return img[iy, ix]
