"""mod_fcg.py - fire control + controls: hammer (cocked, inside the receiver), trigger (traced blade + internal body),
stamped trigger guard, magazine catch, right-side selector lever."""
import numpy as np
import ak_geo as G
from ak_geo import S, R, HW, bp
import details as DT

# trigger-guard centre line read off the left photo (px): front leg x 765, flat bottom row 347, rear leg x 820
TG_PATH = [(765, 304), (765, 336), (766.5, 341), (771, 344), (777, 345.8), (800, 346.8), (812, 346.2), (817, 343.5),
           (819.5, 339), (820.5, 333), (821, 304)]
HAMMER_PIVOT = (766, 296)


def hammer(X, Y, Z):
    """cocked: lies back along the floor, face up under the carrier's rear"""
    yp, zp = bp(*HAMMER_PIVOT)
    d2 = np.minimum(np.hypot(Y - yp, Z - zp) - 5.0, S.capsule2(Y, Z, (yp, zp), (yp - 36, zp + 6), 4.2))
    d2 = np.minimum(d2, S.rect2(Y, Z, (yp - 44, zp + 3), (yp - 32, zp + 12), 1.2))         # head
    d = S.extrude(d2, X, -4.6, 4.6, 0.5)
    return S.diff(d, S.cyl(X, Y, Z, "x", (yp, zp), 2.1, -6, 6))


def trigger(X, Y, Z):
    blade = G.slab(R.prof("trigger", blur=0.3)(Y, Z), X, 3.0, 0.8)
    body = S.extrude(G.poly2(Y, Z, [(782, 304), (812, 304), (816, 296), (790, 292)], 0.6), X, -3.0, 3.0, 0.4)
    return S.runion(blade, body, 0.8)


def tguard(X, Y, Z):
    """stamped trigger guard: a 1.8 mm strip, 13 mm wide, along the measured loop (a continuous band, never traced)"""
    d2 = G.chain2(Y, Z, TG_PATH, 0.9)
    (ya, za), (yb, zb) = bp(819.5, 304), bp(819.5, 336)
    d2 = np.minimum(d2, S.rect2(Y, Z, (min(ya, yb) - 2.2, zb), (max(ya, yb) + 1.6, za), 0.8))   # wide rear tang
    (yc, zc), (yd, zd) = bp(766, 303), bp(819, 307)
    d2 = np.minimum(d2, S.rect2(Y, Z, (min(yc, yd), zd), (max(yc, yd), zc), 0.6))           # top flange under the receiver
    return S.extrude(d2, np.abs(X), -HW["trigger_guard"], HW["trigger_guard"], 0.6)


def magcatch(X, Y, Z):
    return G.slab(R.prof("magcatch", blur=0.4)(Y, Z), X, 7.0, 1.0)


def selector(X, Y, Z):
    """right-side selector: 1.8 mm lever standing off the receiver, rear pivot boss, front finger tab bent outward"""
    x_in = HW["receiver"] + HW["selector_standoff"]
    lever = S.extrude(G.chain2(Y, Z, DT.SEL_PATH, 4.6), -X, x_in, x_in + 1.8, 0.4)
    yp, zp = bp(*DT.SEL_PIVOT)
    boss = S.cyl(X, Y, Z, "x", (yp, zp), 5.6, -(x_in + 2.6), -HW["receiver"] + 0.5, 0.6)
    (ya, za), (yb, zb) = bp(*DT.SEL_TAB[0]), bp(*DT.SEL_TAB[2])
    tab = S.extrude(S.rect2(Y, Z, (min(ya, yb), min(za, zb)), (max(ya, yb), max(za, zb)), 0.8), -X, x_in, x_in + 5.5, 0.5)
    return S.runion(S.runion(lever, boss, 0.6), tab, 0.6)


def _bbox(pts, pad):
    P = np.array([bp(*q) for q in pts])
    return P[:, 0].min() - pad, P[:, 0].max() + pad, P[:, 1].min() - pad, P[:, 1].max() + pad


PARTS = [
    dict(name="hammer", slot="fcg", mat="steel", h=0.2, outer=hammer, box=(G.by(820), G.by(755), -30, 0)),
    dict(name="trigger", slot="fcg", mat="steel", h=0.2, outer=trigger, box=(G.by(820), G.by(778), G.bz(342), G.bz(288))),
    dict(name="tguard", slot="fcg", mat="steel", h=0.2, outer=tguard, trace="tguard"),
    dict(name="magcatch", slot="fcg", mat="steel", h=0.2, outer=magcatch, trace="magcatch"),
    dict(name="selector", slot="selector", mat="steel", h=0.2, outer=selector, box=_bbox(DT.SEL_PATH + DT.SEL_TAB, 8)),
]
