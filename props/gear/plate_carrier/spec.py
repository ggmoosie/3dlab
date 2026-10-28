"""spec.py - plate carrier (TacTec-class layout, NO branding) as softgoods panels. mm, Blender axes: wearer faces -Y,
+X wearer's left, +Z up, sized for the UE5 mannequin (chest ~1.35 m). v0 layout from general knowledge of the design:
shooter's-cut front + back plate bags, padded shoulder straps, PALS cummerbund, loop field, drag handle.
Every number here is layout DATA: a new carrier/vest/rig = a new spec file."""
import numpy as np

COLOR = "#4a503d"                        # ranger green; swap per skin
CHEST = dict(yc=10.0, A=178.0, B=125.0)  # torso ellipse the plate bags sit on (outer surface of the wearer + gap)


def ellipse_arc(A, B, yc, z, a0, a1, n=48, back=False):
    a = np.linspace(a0, a1, n)
    y = yc + (B * np.cos(a) if back else -B * np.cos(a))
    return np.c_[A * np.sin(a), y, np.full(n, z)]


def arclen(P):
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def catmull(pts, n=40):
    P = np.asarray(pts, float); P = np.vstack([P[0], P, P[-1]]); out = []
    for i in range(1, len(P) - 2):
        for t in np.linspace(0, 1, n, endpoint=False):
            p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.vstack([out, P[-2]])


def shooters_cut(w, h, cut_w, cut_h, uc):
    """plate-bag outline centred on uc: square bottom, chamfered top corners"""
    x0, x1 = uc - w / 2, uc + w / 2
    return [(x0, 0), (x1, 0), (x1, h - cut_h), (x1 - cut_w, h), (x0 + cut_w, h), (x0, h - cut_h)]


def panels():
    C = CHEST; out = []
    # ---- front plate bag
    P = ellipse_arc(C["A"], C["B"], C["yc"], 1130, -1.25, 1.25); uc = arclen(P) / 2
    out.append(dict(name="front", path=P, across=(0, 0, 1), center=(0, C["yc"], 1300), t=24, puff=5, puff_w=22, binding=12,
                    color=COLOR, outline=shooters_cut(275, 330, 42, 70, uc), features=[
                        dict(kind="loop", rect=(uc - 95, 248, uc + 95, 300)),
                        dict(kind="molle", rect=(uc - 116, 22, uc + 116, 176), rows=3),
                        dict(kind="seam", pts=[(uc - 125, 205), (uc + 125, 205)], depth=0.6),
                        dict(kind="label", rect=(uc + 90, 186, uc + 118, 198)),
                        dict(kind="wrinkle", rect=(uc - 135, 180, uc - 60, 250), angle=60, amp=0.7, wave=16),
                        dict(kind="wrinkle", rect=(uc + 60, 180, uc + 135, 250), angle=-60, amp=0.7, wave=16)]))
    # ---- back plate bag
    P = ellipse_arc(C["A"], C["B"], C["yc"], 1110, -1.25, 1.25, back=True); uc = arclen(P) / 2
    out.append(dict(name="back", path=P, across=(0, 0, 1), center=(0, C["yc"], 1300), t=24, puff=5, puff_w=22, binding=12,
                    color=COLOR, outline=shooters_cut(280, 350, 40, 66, uc), features=[
                        dict(kind="molle", rect=(uc - 118, 20, uc + 118, 280), rows=5),
                        dict(kind="wrinkle", rect=(uc - 140, 280, uc + 140, 350), angle=90, amp=0.6, wave=18)]))
    # ---- cummerbund, both sides (under the front bag, PALS 3 rows)
    for sx in (-1, 1):
        a = np.linspace(0, 1, 60)
        ang = (0.95 + (2.35 - 0.95) * a) * sx                         # from the back edge round the side to the front
        Ac, Bc = C["A"] - 10, C["B"] - 8
        P = np.c_[Ac * np.sin(ang), C["yc"] + Bc * np.cos(ang), np.full(60, 1150)]
        L = arclen(P)
        out.append(dict(name=f"cummerbund_{'L' if sx > 0 else 'R'}", path=P, across=(0, 0, 1), center=(0, C["yc"], 1300),
                        t=6, puff=1.5, puff_w=10, binding=10, color=COLOR, outline=[(4, 0), (L - 4, 0), (L - 4, 160), (4, 160)],
                        features=[dict(kind="molle", rect=(14, 4, L - 30, 156), rows=3),
                                  dict(kind="wrinkle", rect=(L - 110, 0, L - 4, 160), angle=80, amp=1.0, wave=12),
                                  dict(kind="seam", pts=[(L - 26, 8), (L - 26, 152)], depth=0.5)]))
    # ---- padded shoulder straps
    for sx in (-1, 1):
        x = 96 * sx
        P = catmull([(x, -108, 1420), (x * 1.01, -98, 1478), (x * 1.04, -62, 1536), (x * 1.05, -5, 1556),
                     (x * 1.04, 52, 1540), (x * 1.01, 102, 1492), (x, 118, 1440)])
        L = arclen(P)
        out.append(dict(name=f"strap_{'L' if sx > 0 else 'R'}", path=P, across=(1, 0, 0), center=(x, 10, 1380), t=12, puff=4,
                        puff_w=14, binding=8, color=COLOR, outline=[(0, -32), (L, -32), (L, 32), (0, 32)],
                        features=[dict(kind="stitch", pts=[(10, 0), (L - 10, 0)]),
                                  dict(kind="molle", rect=(L * 0.18, -13, L * 0.42, 13), rows=1),
                                  dict(kind="wrinkle", rect=(L * 0.35, -32, L * 0.65, 32), angle=0, amp=0.8, wave=10)]))
    # ---- drag handle (webbing loop on the back bag's top edge)
    P = catmull([(-42, 120, 1452), (-30, 128, 1478), (0, 132, 1488), (30, 128, 1478), (42, 120, 1452)])
    out.append(dict(name="drag_handle", path=P, across=(0, 1, 0), center=(0, 125, 1420), t=3, puff=0.6, puff_w=6, binding=0,
                    color=COLOR, outline=[(0, -19), (arclen(P), -19), (arclen(P), 19), (0, 19)],
                    features=[dict(kind="molle", rect=(-5, -19, arclen(P) + 5, 19), rows=1, web=38)]))
    return out
