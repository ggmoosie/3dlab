"""Rig + proxy body + tailoring fields for the hoodie (scene units: 1 BU = 1 cm, scale_length 0.01, like the pill-bottle clip).

- rig(): the export armature. Bones come from anim/pill_bottle/pill_bottle_anim.build_rig (json heads, chain tails, roll 0),
  plus the contract root bone `root` at the origin as the parent of pelvis.
- weight_rig(): a throwaway copy whose bone SEGMENTS lie along the limbs (twist bones own their third of the arm / leg), so
  heat weighting has sensible segments. Same names and heads; only the tails differ. Vertex groups transfer by name.
- body_sdf / garment_sdf: analytic signed distance fields (numpy, (N,3) cm) of the proxy body and of the hoodie's outer
  surface (body + ease + fabric thickness). The proxy mesh is marching cubes of body_sdf; it is never exported.
"""
import os, sys, math, json, importlib.util
import numpy as np
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
SKEL = "/home/user/3d/props/brief/skeleton_ue5_manny_rest.json"


def _pill():
    spec = importlib.util.spec_from_file_location("pill_bottle_anim", os.path.join(ROOT, "anim/pill_bottle/pill_bottle_anim.py"))
    m = importlib.util.module_from_spec(spec)
    argv = sys.argv; sys.argv = [argv[0]]                     # its module-level ARGS parse must not see our args
    try:
        spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


PILL = _pill()
D = json.load(open(SKEL))["bones"]


def ue(v):
    return np.array([v[0], -v[1], v[2]], np.float64)


J = {n: ue(b["head"]) for n, b in D.items()}                  # joint positions, Blender cm


def rig(name="SK_Hoodie_Rig"):
    """Export armature: pill-bottle build_rig + root bone. Object named 'Armature' so UE does not add a node bone."""
    r, _ = PILL.build_rig()
    arm = r.data
    bpy.context.view_layer.objects.active = r
    bpy.ops.object.mode_set(mode="EDIT")
    e = arm.edit_bones.new("root")
    e.head, e.tail, e.roll = Vector((0, 0, 0)), Vector((0, -15, 0)), 0.0          # UE root: at the floor, +Y (UE) forward
    arm.edit_bones["pelvis"].parent = e
    arm.edit_bones["pelvis"].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.bones["root"].use_deform = False
    r.name, arm.name = "Armature", name
    return r


SEG = {  # weighting segments (start joint, end point) for heat weights
    "upperarm_{s}": ("upperarm_{s}", ("upperarm_twist_01_{s}",)),
    "upperarm_twist_01_{s}": ("upperarm_twist_01_{s}", ("upperarm_twist_02_{s}",)),
    "upperarm_twist_02_{s}": ("upperarm_twist_02_{s}", ("lowerarm_{s}",)),
    "lowerarm_{s}": ("lowerarm_{s}", ("lowerarm_twist_02_{s}",)),
    "lowerarm_twist_02_{s}": ("lowerarm_twist_02_{s}", ("lowerarm_twist_01_{s}",)),
    "lowerarm_twist_01_{s}": ("lowerarm_twist_01_{s}", ("hand_{s}",)),
    "thigh_{s}": ("thigh_{s}", ("thigh_twist_01_{s}",)),
    "thigh_twist_01_{s}": ("thigh_twist_01_{s}", ("calf_{s}",)),
    "calf_{s}": ("calf_{s}", ("calf_twist_01_{s}",)),
    "calf_twist_01_{s}": ("calf_twist_01_{s}", ("foot_{s}",)),
    "foot_{s}": ("foot_{s}", ("ball_{s}",)),
    "clavicle_{s}": ("clavicle_{s}", ("upperarm_{s}",)),
}


def seg_tail(n):
    """Tail used for weighting: along the limb to the next joint (twist bones own their third)."""
    for k, (a, (b,)) in SEG.items():
        for s in "lr":
            if k.format(s=s) == n:
                return J[b.format(s=s)]
    chain = {"pelvis": "spine_01", "spine_01": "spine_02", "spine_02": "spine_03", "spine_03": "spine_04", "spine_04": "spine_05",
             "spine_05": "neck_01", "neck_01": "neck_02", "neck_02": "head"}
    if n in chain:
        return J[chain[n]]
    if n == "head":
        return J["head"] + np.array([0, 0, 16.0])
    if n.startswith("hand"):
        s = n[-1]
        return J[n] + (J[n] - J["lowerarm_" + s]) / np.linalg.norm(J[n] - J["lowerarm_" + s]) * 9
    if n.startswith("ball"):
        return J[n] + np.array([0, -6.0, 0])
    raise KeyError(n)


def weight_rig():
    arm = bpy.data.armatures.new("WeightRig")
    r = bpy.data.objects.new("WeightRig", arm)
    bpy.context.scene.collection.objects.link(r)
    bpy.context.view_layer.objects.active = r
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for n, b in D.items():
        e = arm.edit_bones.new(n)
        e.head, e.tail = Vector(J[n]), Vector(seg_tail(n))
        eb[n] = e
    for n, b in D.items():
        if b["parent"]:
            eb[n].parent = eb[b["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    return r


# ----------------------------------------------------------------------------------------------- SDF primitives (numpy)
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def cone(P, a, b, r1, r2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    t = np.clip(((P - a) @ ba) / (ba @ ba), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ba), axis=1) - (r1 + (r2 - r1) * t)


def cone_t(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    return ((P - a) @ ba) / (ba @ ba)


def ellipsoid(P, c, r):
    q = (P - np.asarray(c, float)) / np.asarray(r, float)
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / np.asarray(r, float), axis=1)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-9)


# torso loft: z, half-width a (x), half-depth b (y), centre y (front = -Y)
BODY_T = np.array([
    (82, 16.6, 10.8, -1.4), (88, 17.4, 11.3, -1.8), (95, 17.1, 11.4, -2.8), (103, 15.7, 10.6, -4.6), (111, 15.0, 10.4, -5.6),
    (119, 15.7, 10.9, -6.0), (127, 16.7, 11.7, -6.0), (135, 17.2, 11.5, -5.5), (141, 16.3, 10.2, -3.9), (146, 12.6, 8.4, -1.6),
    (150, 7.6, 6.8, 0.2)])
# outer-surface offset of the hoodie over the body loft (gap + fabric): band 0.5+0.7 rib, chest 0.8+0.45, waist relaxed
EASE_T = np.array([(84.5, 1.25), (91.5, 1.3), (94.5, 2.6), (103, 2.3), (111, 2.1), (119, 1.7), (126, 1.35), (135, 1.25),
                   (141, 1.25), (146, 1.3), (151, 1.3)])
Z_HEM, Z_BAND = 84.5, 91.5


def loft(P, tab, ease=None, z0=None, z1=None):
    z = P[:, 2]
    a = np.interp(z, tab[:, 0], tab[:, 1]); b = np.interp(z, tab[:, 0], tab[:, 2]); c = np.interp(z, tab[:, 0], tab[:, 3])
    if ease is not None:
        e = np.interp(z, ease[:, 0], ease[:, 1]); a, b = a + e, b + e
    q = np.stack([P[:, 0] / a, (P[:, 1] - c) / b], 1)
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / np.stack([a, b], 1), axis=1)
    d = k0 * (k0 - 1) / np.maximum(k1, 1e-9)
    lo, hi = (tab[0, 0] if z0 is None else z0), (tab[-1, 0] if z1 is None else z1)
    cap = np.maximum(lo - z, z - hi)
    return np.maximum(d, cap)


def _side(s):
    return 1.0 if s == "l" else -1.0


def limbs(P, inflate=0.0, legs=True):
    """Arms, shoulders, neck, head, legs (body). inflate adds to every radius."""
    d = np.full(len(P), 1e9)
    for s in "lr":
        x = _side(s)
        sh, el, wr = J["upperarm_" + s], J["lowerarm_" + s], J["hand_" + s]
        fdir = (wr - el) / np.linalg.norm(wr - el)
        d = np.minimum(d, smin(cone(P, (x * 5, 1.0, 148.0), (x * 16.5, 2.4, 145.6), 5.0, 4.6),
                               ellipsoid(P, (x * 19.0, 2.4, 142.2), (6.0, 6.3, 7.2)), 3.0))
        d = smin(d, cone(P, sh, el, 5.0, 4.0), 2.0)
        d = smin(d, cone(P, el, wr, 4.1, 2.9), 1.5)
        d = smin(d, cone(P, wr, wr + fdir * 8.5, 2.6, 2.2), 1.0)
        if legs:
            th, kn, an, ba = J["thigh_" + s], J["calf_" + s], J["foot_" + s], J["ball_" + s]
            d = np.minimum(d, smin(cone(P, th, kn, 8.6, 5.6), cone(P, kn, an, 5.4, 3.4), 2.0))
            d = smin(d, ellipsoid(P, (x * 12.8, 0.5, 37), (5.4, 5.8, 9.0)), 2.0)
            d = smin(d, cone(P, an + np.array([0, 1.0, -2.5]), ba + np.array([0, -2.0, 1.2]), 3.6, 2.9), 2.0)
    neck = cone(P, (0, 1.2, 146.0), (0, 0.8, 160.0), 6.0, 5.5)
    head = smin(ellipsoid(P, (0, -1.0, 169.0), (8.0, 9.8, 11.5)), ellipsoid(P, (0, -4.6, 162.0), (6.0, 6.0, 5.0)), 2.0)
    return d, smin(neck, head, 3.0)


def body_sdf(P):
    P = np.asarray(P, float)
    d, nh = limbs(P)
    t = loft(P, BODY_T)
    return smin(smin(t, d, 2.5), nh, 2.5)


def arm_garment(P, s):
    """Sleeve outer surface around one arm: relaxed (2 cm gap + fabric) on the arm, ribbed cuff (0.5 gap + 0.7 rib) at the wrist."""
    sh, el, wr = J["upperarm_" + s], J["lowerarm_" + s], J["hand_" + s]
    fl = np.linalg.norm(wr - el)
    t = cone_t(P, el, wr)                                     # 0 elbow .. 1 wrist
    cuff_top = 1 - 6.5 / fl
    # forearm radius: body 4.1 -> 2.9; outer sleeve: + 2.45 relaxed, bunching flare just above the cuff, + 1.2 at the cuff
    rb = 4.1 + (2.9 - 4.1) * np.clip(t, 0, 1)
    ease = np.interp(t, [0, 0.45, cuff_top - 0.06, cuff_top, cuff_top + 0.05, 1.0], [2.45, 2.5, 2.7, 1.9, 1.25, 1.2])
    fore = cone(P, el, wr + (wr - el) / fl * 0.6, 1.0, 1.0) + 1.0 - (rb + ease)
    up = cone(P, sh, el, 5.0 + 2.45, 4.1 + 2.45)
    return smin(up, fore, 1.5)


def core_body(P):
    """Torso, shoulders, arms to the wrist and the thigh tops (no neck / head / hands): what the hoodie must clear."""
    P = np.asarray(P, float)
    d = loft(P, BODY_T)
    for s in "lr":
        x = _side(s)
        sh, el, wr = J["upperarm_" + s], J["lowerarm_" + s], J["hand_" + s]
        d = smin(d, smin(cone(P, (x * 5, 1.0, 148.0), (x * 16.5, 2.4, 145.6), 5.0, 4.6),
                         ellipsoid(P, (x * 19.0, 2.4, 142.2), (6.0, 6.3, 7.2)), 3.0), 2.5)
        d = smin(d, cone(P, sh, el, 5.0, 4.0), 2.0)
        d = smin(d, cone(P, el, wr, 4.1, 2.9), 1.5)
        th, kn = J["thigh_" + s], J["calf_" + s]
        d = smin(d, np.maximum(cone(P, th, kn, 8.6, 5.6), 80.0 - P[:, 2]), 2.5)
    return smin(d, cone(P, (0, 1.2, 146.0), (0, 0.8, 160.0), 6.0, 5.5), 2.5)


MIN_OUTER = 1.05          # outer surface never closer than this to the core body (fabric 0.45-0.7 + >= 0.35 gap)


def garment_sdf(P):
    """Outer surface of the hoodie body + sleeves (hood and pocket are built on top of it)."""
    P = np.asarray(P, float)
    return np.minimum(_garment(P), core_body(P) - MIN_OUTER)


def _garment(P):
    t = loft(P, BODY_T, EASE_T, z0=Z_HEM - 3, z1=150.5)
    sh = np.full(len(P), 1e9)
    for s in "lr":
        x = _side(s)
        shoulder = smin(cone(P, (x * 5, 1.0, 148.0), (x * 16.5, 2.4, 145.6), 5.0 + 1.3, 4.6 + 1.3),
                        ellipsoid(P, (x * 19.0, 2.4, 142.2), (6.0 + 1.5, 6.3 + 1.5, 7.2 + 1.5)), 3.0)
        sh = np.minimum(sh, smin(shoulder, arm_garment(P, s), 3.0))
    return smin(t, sh, 4.0)


def neck_sdf(P):
    _, nh = limbs(np.asarray(P, float), legs=False)
    return nh


def grad(f, P, h=0.05):
    P = np.asarray(P, float)
    g = np.zeros_like(P)
    for i in range(3):
        e = np.zeros(3); e[i] = h
        g[:, i] = (f(P + e) - f(P - e)) / (2 * h)
    return g


def project(f, P, target=0.0, iters=8, step=1.0):
    """Move points along the field gradient onto f = target."""
    P = np.array(P, float)
    for _ in range(iters):
        v = f(P) - target
        g = grad(f, P)
        n2 = np.maximum((g * g).sum(1), 1e-9)
        P -= step * (v / n2)[:, None] * g
    return P


def proxy_mesh(name="Proxy_Body", res=0.9):
    """Marching cubes of body_sdf -> one closed mesh (never exported)."""
    from skimage import measure
    lo, hi = np.array([-58, -30, -2.0]), np.array([58, 22, 183.0])
    n = np.ceil((hi - lo) / res).astype(int) + 1
    xs, ys, zs = (np.linspace(lo[i], lo[i] + (n[i] - 1) * res, n[i]) for i in range(3))
    vol = np.empty(n, np.float32)
    for i, x in enumerate(xs):
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        P = np.stack([np.full(Y.size, x), Y.ravel(), Z.ravel()], 1)
        vol[i] = body_sdf(P).reshape(Y.shape)
    v, f, _, _ = measure.marching_cubes(vol, 0.0, spacing=(res, res, res))
    v += lo
    c = v.mean(0)                                             # outward winding: positive signed volume
    vol = np.einsum("ij,ij->i", v[f[:, 0]] - c, np.cross(v[f[:, 1]] - c, v[f[:, 2]] - c)).sum()
    if vol < 0:
        f = f[:, ::-1]
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.tolist(), [], f.tolist())
    me.validate()
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    for p in me.polygons:
        p.use_smooth = True
    return o
