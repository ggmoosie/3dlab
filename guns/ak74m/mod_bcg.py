"""mod_bcg.py - bolt carrier group, modelled in battery: carrier (body + front bridge + long-stroke piston + charging
handle on the right), rotating bolt (2 lugs, cam pin, recessed face), recoil assembly (guide rod, coil spring, rear
block with the dust-cover latch button). Real-world layout fitted inside iface's receiver cavity; animates along -y by
iface.CARRIER_TRAVEL."""
import numpy as np
import ak_geo as G
from ak_geo import S
from iface import Y as IY, Z as IZ

BF = IY["bolt_face"]
C_REAR = BF - 136.0                     # carrier body rear
C_FRONT = IY["trunnion_front"] - 3.0    # front bridge (over the trunnion)
HANDLE_Y = G.by(633)


def carrier(X, Y, Z):
    body = S.box(X, Y, Z, (-11.5, C_REAR, -1.0), (11.5, BF - 2.0, 33.0), 2.0)
    bridge = S.box(X, Y, Z, (-9.0, BF - 4.0, 13.0), (9.0, C_FRONT, 33.0), 2.0)
    d = S.runion(body, bridge, 2.0)
    d = S.rdiff(d, S.cyl(X, Y, Z, "y", (0, 0), 9.4, BF - 75, BF + 5), 0.5)               # bolt channel
    for sx in (-1, 1):                                                                     # rail grooves both sides
        d = S.rdiff(d, np.maximum.reduce([sx * X - 10.6, np.abs(Z - 16.0) - 1.6, G.yband(Y, C_REAR - 1, BF + 1)]), 0.3)
    cam = np.maximum.reduce([X + 0.0, np.abs((Y - (BF - 30)) * 0.5 + (Z - 9)) - 3.0, G.yband(Y, BF - 48, BF - 14), np.abs(Z - 9) - 9])
    d = S.rdiff(d, cam, 0.4)                                                               # cam track (right)
    d = S.rdiff(d, S.cyl(X, Y, Z, "y", (0, IZ["guide"]), 5.4, C_REAR - 2, C_REAR + 46), 0.4)  # spring channel
    d = S.rdiff(d, S.box(X, Y, Z, (-6, C_REAR - 1, -2), (6, C_REAR + 40, 6), 1.0), 0.6)  # hammer clearance
    piston = np.minimum(S.cyl(X, Y, Z, "y", (0, IZ["piston"]), 5.2, C_FRONT - 2, -206.0, 0.6),
                        S.cyl(X, Y, Z, "y", (0, IZ["piston"]), 7.6, -206.0, -191.0, 0.8))
    for yg in (-196.0, -201.0):
        piston = S.rdiff(piston, np.maximum(np.abs(Y - yg) - 0.6, 6.8 - np.hypot(X, Z - IZ["piston"])), 0.2)
    d = S.runion(d, piston, 1.5)
    arm = S.box(X, Y, Z, (-26.0, HANDLE_Y - 4.0, IZ["handle"] - 2.0), (-8.0, HANDLE_Y + 4.0, IZ["handle"] + 2.0), 1.0)
    knob = S.cyl(X, Y, Z, "x", (HANDLE_Y, IZ["handle"]), 5.2, -31.5, -23.0, 1.6)
    return S.runion(S.runion(d, arm, 1.5), knob, 1.2)


def bolt(X, Y, Z):
    d = S.cyl(X, Y, Z, "y", (0, 0), 8.8, BF - 64, BF, 0.6)
    for sx in (-1, 1):
        d = S.runion(d, S.box(X, Y, Z, (min(0, sx * 12.2), BF - 9, -3.2), (max(0, sx * 12.2), BF, 3.2), 0.8), 0.6)   # locking lugs
    d = S.runion(d, S.cyl(X, Y, Z, "z", (0, BF - 30), 2.8, 0, 13.5, 0.5), 0.6)              # cam pin
    d = S.rdiff(d, S.cyl(X, Y, Z, "y", (0, 0), 5.1, BF - 1.4, BF + 1), 0.3)                 # case-head recess
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 1.0, BF - 70, BF + 1))                         # firing-pin hole
    return S.rdiff(d, S.box(X, Y, Z, (-10, BF - 9, 4.2), (-6.5, BF + 1, 7.0), 0.3), 0.2)  # extractor slot (right)


def recoil(X, Y, Z):
    zg = IZ["guide"]
    rodd = S.cyl(X, Y, Z, "y", (0, zg), 1.8, IY["receiver_rear"] + 12, C_REAR + 30, 0.4)
    spring = G.helix(X, Y, Z, zg, 4.2, 0.75, 4.6, C_REAR + 26, IY["receiver_rear"] + 14)
    blk = S.box(X, Y, Z, (-7.0, IY["receiver_rear"] + 1.5, zg - 6), (7.0, IY["receiver_rear"] + 14, zg + 4), 1.0)
    btn = S.box(X, Y, Z, (-3.4, IY["receiver_rear"] + 2.5, zg), (3.4, IY["receiver_rear"] + 8.5, IZ["guide_button"] + 4.5), 0.8)
    return np.minimum(np.minimum(rodd, spring), S.runion(blk, btn, 0.8))


PARTS = [
    dict(name="carrier", slot="bcg", mat="steel", h=0.25, outer=carrier, box=(C_REAR - 2, -188.0, -4, 36), xh=34),
    dict(name="bolt", slot="bcg", mat="steel", h=0.2, outer=bolt, box=(BF - 72, BF + 2, -15, 16)),
    dict(name="recoil", slot="recoil", mat="steel", h=0.2, outer=recoil, box=(IY["receiver_rear"], C_REAR + 32, 16, 40)),
]
