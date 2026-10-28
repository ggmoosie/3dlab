"""mod_receiver.py - stamped receiver as a real 1 mm shell (open top with inward rails, magwell + trigger slot through
the floor, right-side ejection port, solid front/rear trunnions), through pins, rear sight (block, leaf, slider,
gas-tube lever), dust cover as a 0.9 mm shell (open front + bottom, handle-travel cut right, latch hole rear)."""
import numpy as np
import ak_geo as G
from ak_geo import S, R, HW, bp
import details as DT
from iface import Y as IY, Z as IZ, REC_T, COVER_T

HWR = HW["receiver"]


def receiver(X, Y, Z):
    d = G.slab(R.prof("receiver")(Y, Z), X, HWR, 1.6)
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 12.0, IY["bolt_face"] - 0.1, IY["trunnion_front"] + 2))   # barrel / bolt-lug bore


def receiver_cavity(X, Y, Z):
    P = R.prof("receiver")
    inner_hw = np.abs(X) - (HWR - REC_T)
    floor = IZ["receiver_floor"] + REC_T - Z
    body = np.maximum.reduce([P(Y, np.minimum(Z, 8.0)) + REC_T, inner_hw, floor, G.yband(Y, IY["bolt_face"], IY["rear_trunnion_front"])])
    rails = np.maximum.reduce([np.abs(X) - (HWR - 3.4), 10.0 - Z, G.yband(Y, IY["trunnion_front"] - 2, IY["receiver_rear"] + 8)])   # open top between the rails
    over_trunnion = np.maximum.reduce([inner_hw, 12.0 - Z, G.yband(Y, IY["trunnion_front"] - 2, IY["bolt_face"])])
    magwell = np.maximum.reduce([np.abs(X) - 13.6, G.yband(Y, IY["magwell_front"], IY["magwell_rear"]), Z - (IZ["receiver_floor"] + 3)])
    tslot = np.maximum.reduce([np.abs(X) - 4.5, G.yband(Y, IY["trigger_slot_front"], IY["trigger_slot_rear"]), Z - (IZ["receiver_floor"] + 3)])
    port = np.maximum.reduce([X + 9.0, G.yband(Y, IY["port_front"], IY["port_rear"]), 9.0 - Z])            # right wall only
    return np.minimum.reduce([body, rails, over_trunnion, magwell, tslot, port])


PINS = [(751, 302), (766, 298), (808, 307)]


def pins(X, Y, Z):
    d = np.full(X.shape, 1e3)
    for q in PINS:
        y, z = bp(*q)
        d = np.minimum(d, S.cyl(X, Y, Z, "x", (y, z), 2.0, -HWR - 0.2, HWR + 0.2, 0.3))
    y, z = bp(*DT.SEL_PIVOT)
    return np.minimum(d, S.cyl(X, Y, Z, "x", (y, z), 3.0, -HWR - 0.2, HWR - REC_T + 0.5, 0.3))   # selector shaft


def rsight(X, Y, Z):
    P2 = R.prof("rsight")(Y, Z)
    zl = G.bz(236)
    hw = np.where(Z > zl, HW["rear_leaf"], HW["rear_sight_block"])
    d = G.slab(P2, X, hw, 1.0)
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, IZ["piston"]), 8.2, IY["rsight_rear"] - 2, IY["rsight_front"] + 2))   # piston passes


def rs_slider(X, Y, Z):
    """range slider riding the leaf + its two side catch buttons"""
    (y0, z0), (y1, z1) = bp(597, 224), bp(611, 236)
    body = S.box(X, Y, Z, (-HW["rear_leaf"] - 1.6, min(y0, y1), min(z0, z1)), (HW["rear_leaf"] + 1.6, max(y0, y1), max(z0, z1)), 0.6)
    yc, zc = (y0 + y1) / 2, (z0 + z1) / 2
    btn = S.cyl(X, Y, Z, "x", (yc, zc), 2.3, -HW["rear_leaf"] - 4.2, HW["rear_leaf"] + 4.2, 0.5)
    return S.runion(body, btn, 0.4)


def gt_lever(X, Y, Z):
    """gas-tube lock lever, right side of the rear sight block"""
    x_in = HW["rear_sight_block"]
    d2 = G.chain2(Y, Z, [(583, 236), (595, 248)], 2.6)
    lever = S.extrude(d2, -X, x_in - 0.3, x_in + 2.0, 0.4)
    y, z = bp(595, 248)
    return S.runion(lever, S.cyl(X, Y, Z, "x", (y, z), 3.6, -(x_in + 2.8), -(x_in - 0.5), 0.5), 0.5)


def _cover_top(X, Y):
    from scipy.ndimage import median_filter, gaussian_filter1d
    cr = R.column_radius(*G.xr("dcover"), "dcover")
    ys = np.array([c[0] for c in cr]); o = np.argsort(ys)
    zt = gaussian_filter1d(median_filter(np.array([c[2] + c[1] for c in cr]), 9, mode="nearest"), 3, mode="nearest")
    return np.interp(Y, ys[o], zt[o])


def dcover(X, Y, Z):
    """D-section: vertical sides, round top (radius = half width)"""
    hw = HW["dust_cover"]
    top = np.hypot(X, np.maximum(Z - (_cover_top(X, Y) - hw), 0)) - hw
    return S.rinter(R.prof("dcover")(Y, Z), np.maximum(np.abs(X) - hw, top), 0.6)


def dcover_cavity(X, Y, Z):
    hw = HW["dust_cover"] - COVER_T
    top = np.hypot(X, np.maximum(Z - (_cover_top(X, Y) - COVER_T - hw), 0)) - hw
    shell = np.maximum.reduce([np.abs(X) - hw, top, G.yband(Y, IY["rsight_rear"] - 3, IY["receiver_rear"] + 4.0)])
    handle = np.maximum.reduce([X + 12.0, G.yband(Y, IY["port_front"] + 14, IY["port_rear"] - 110), Z - 25.0])   # handle travel, right
    latch = S.box(X, Y, Z, (-4.2, IY["receiver_rear"] - 4, IZ["guide_button"] - 4), (4.2, IY["receiver_rear"] + 10, IZ["guide_button"] + 4), 0.6)
    return np.minimum.reduce([shell, handle, latch])


PARTS = [
    dict(name="receiver", slot="receiver", mat="steel", h=0.3, outer=receiver, inner=receiver_cavity, trace="receiver"),
    dict(name="pins", slot="fcg", mat="steel", h=0.2, outer=pins, box=(G.by(840), G.by(745), -32, 6)),
    dict(name="rsight", slot="rear_sight", mat="park", h=0.25, outer=rsight, trace="rsight"),
    dict(name="rs_slider", slot="rear_sight", mat="steel", h=0.2, outer=rs_slider, box=(G.by(615), G.by(593), G.bz(240), G.bz(220))),
    dict(name="gt_lever", slot="upper_handguard", mat="steel", h=0.2, outer=gt_lever, box=(G.by(602), G.by(577), G.bz(256), G.bz(229))),
    dict(name="dcover", slot="dust_cover", mat="steel", h=0.25, outer=dcover, inner=dcover_cavity, trace="dcover"),
]
