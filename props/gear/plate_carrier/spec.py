"""spec.py - plate carrier (TacTec-class layout, NO branding) as softgoods panels. mm, Blender axes: wearer faces -Y,
+X wearer's left, +Z up, sized for the UE5 mannequin (chest ~1.35 m). v0 layout from general knowledge of the design:
shooter's-cut front + back plate bags, padded shoulder straps, PALS cummerbund, loop field, drag handle.
Every number here is layout DATA: a new carrier/vest/rig = a new spec file."""
import numpy as np
from lab.softgoods.core import rrect

COLOR = "#2b2c2e"                        # black (refs); ranger green "#4a503d", coyote "#7a6a4f"
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
    """TacTec-class layout read off the user's references (front 3/4, side, back, front on mannequin); no logo patch"""
    C = CHEST; out = []
    # ---- front plate bag: loop strip, pull tab, laser-cut PALS grid; lower third hidden by the front flap
    P = ellipse_arc(C["A"], C["B"], C["yc"], 1130, -1.25, 1.25); uc = arclen(P) / 2
    out.append(dict(name="front", path=P, across=(0, 0, 1), center=(0, C["yc"], 1300), t=24, puff=5, puff_w=22, binding=10,
                    color=COLOR, outline=shooters_cut(275, 335, 38, 62, uc), features=[
                        dict(kind="loop", rect=(uc - 88, 296, uc + 88, 324)),
                        dict(kind="label", rect=(uc - 16, 283, uc + 16, 291)),
                        dict(kind="lasercut", rect=(uc - 118, 102, uc + 118, 280)),
                        dict(kind="wrinkle", rect=(uc - 135, 230, uc - 70, 330), angle=60, amp=0.6, wave=16),
                        dict(kind="wrinkle", rect=(uc + 70, 230, uc + 135, 330), angle=-60, amp=0.6, wave=16)]))
    # ---- back plate bag: handle/loop strip, laser-cut grid
    P = ellipse_arc(C["A"], C["B"], C["yc"], 1110, -1.25, 1.25, back=True); uc = arclen(P) / 2
    out.append(dict(name="back", path=P, across=(0, 0, 1), center=(0, C["yc"], 1300), t=24, puff=5, puff_w=22, binding=10,
                    color=COLOR, outline=shooters_cut(280, 350, 38, 62, uc), features=[
                        dict(kind="loop", rect=(uc - 60, 312, uc + 60, 338)),
                        dict(kind="lasercut", rect=(uc - 118, 100, uc + 118, 300)),
                        dict(kind="wrinkle", rect=(uc - 140, 300, uc + 140, 350), angle=90, amp=0.5, wave=18)]))
    # ---- full-width front + back flaps: 3 rows of real PALS webbing (the cummerbund closes under them)
    for nm, back, z in (("flap_front", False, 1118), ("flap_back", True, 1100)):
        P = ellipse_arc(C["A"] + 33, C["B"] + 33, C["yc"], z, -1.05, 1.05, back=back); uc = arclen(P) / 2
        out.append(dict(name=nm, path=P, across=(0, 0, 1), center=(0, C["yc"], 1300), t=7, puff=2.5, puff_w=12, binding=9,
                        color=COLOR, outline=rrect(uc - 178, 0, uc + 178, 150, r=10), features=[
                            dict(kind="molle", rect=(uc - 168, 4, uc + 168, 148), rows=3),
                            dict(kind="wrinkle", rect=(uc - 178, 0, uc - 90, 150), angle=75, amp=0.7, wave=14),
                            dict(kind="wrinkle", rect=(uc + 90, 0, uc + 178, 150), angle=-75, amp=0.7, wave=14)]))
    # ---- elastic sides: ruched bungee bands + webbing ladder, gaps between
    for sx in (-1, 1):
        ang = (0.85 + (2.20 - 0.85) * np.linspace(0, 1, 60)) * sx
        Ac, Bc = C["A"] - 4, C["B"] - 3
        P = np.c_[Ac * np.sin(ang), C["yc"] + Bc * np.cos(ang), np.full(60, 1122)]
        L = arclen(P)
        out.append(dict(name=f"side_{'L' if sx > 0 else 'R'}", path=P, across=(0, 0, 1), center=(0, C["yc"], 1300), t=5, puff=1.0,
                        puff_w=8, binding=0, color=COLOR, outline=[(2, 0), (L - 2, 0), (L - 2, 140), (2, 140)],
                        features=[dict(kind="elastic", rect=(2, 0, L - 2, 140), rows=3)]))
    # ---- wide padded shoulder pads (spacer mesh underneath via the inner tile), adjuster cover tab at the front
    for sx in (-1, 1):
        x = 98 * sx
        P = catmull([(x, -112, 1418), (x * 1.01, -104, 1470), (x * 1.04, -72, 1528), (x * 1.06, -15, 1552),
                     (x * 1.05, 45, 1540), (x * 1.02, 98, 1494), (x, 120, 1440)])
        L = arclen(P)
        out.append(dict(name=f"pad_{'L' if sx > 0 else 'R'}", path=P, across=(1, 0, 0), center=(x, 10, 1380), t=16, puff=6,
                        puff_w=18, binding=8, color=COLOR, outline=rrect(0, -48, L, 48, r=14),
                        features=[dict(kind="pocket", rect=(8, -34, L * 0.24, 34), flap=14, puff=2.5),
                                  dict(kind="stitch", pts=[(L * 0.28, -30), (L * 0.95, -30)]),
                                  dict(kind="stitch", pts=[(L * 0.28, 30), (L * 0.95, 30)]),
                                  dict(kind="wrinkle", rect=(L * 0.3, -48, L * 0.7, 48), angle=0, amp=0.9, wave=11)]))
    # ---- drag handle (webbing loop on the back bag's top edge)
    P = catmull([(-42, 122, 1456), (-30, 130, 1482), (0, 134, 1492), (30, 130, 1482), (42, 122, 1456)])
    out.append(dict(name="drag_handle", path=P, across=(0, 1, 0), center=(0, 125, 1420), t=3, puff=0.6, puff_w=6, binding=0,
                    color=COLOR, outline=[(0, -19), (arclen(P), -19), (arclen(P), 19), (0, 19)],
                    features=[dict(kind="molle", rect=(-5, -19, arclen(P) + 5, 19), rows=1, web=38)]))
    return out
