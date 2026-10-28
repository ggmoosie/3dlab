"""mod_handguard.py - gas tube + upper handguard (one swappable unit, like the real part), lower handguard (polymer
shell, open top, barrel channel), handguard retainer cap (steel band). Replacements (Zenit B-10 / B-11 class) fill the
same slots and stay inside iface.SLOTS[...]['envelope']."""
import numpy as np
import ak_geo as G
from ak_geo import S, R, HW
from iface import Y as IY, Z as IZ, POLY_T


def gastube(X, Y, Z):
    """photo-lathed visible front + hidden run to the rear sight block; hollow (piston inside), 2 vents each side"""
    cr = R.column_radius(*G.xr("gastube"), "gastube")
    zc = float(np.median([c[2] for c in cr]))
    d = G.lathe_photo(Y, Z, X, "gastube", zc=zc, rmax=HW["gas_tube"] + 1.5, smooth=3)
    d = np.minimum(d, S.cyl(X, Y, Z, "y", (0, IZ["piston"]), IZ["gastube_r"], IY["rsight_front"] - 4, G.by(446)))
    d = S.diff(d, S.cyl(X, Y, Z, "y", (0, IZ["piston"]), IZ["gastube_r"] - 1.2, IY["rsight_front"] - 6, IY["gastube_front"] + 2))
    for yv in (-218.0, -226.0):
        d = S.diff(d, S.cyl(X, Y, Z, "x", (yv, IZ["piston"]), 1.6, -20, 20))
    return d


def hg_upper(X, Y, Z):
    d = G.slab(R.prof("hg_upper")(Y, Z), X, HW["hg_upper"], 8.0)
    return S.rdiff(d, S.cyl(X, Y, Z, "y", (0, IZ["piston"]), IZ["gastube_r"] + 0.4, IY["rsight_front"] - 20, 0), 0.4)


def _hgl_outer(X, Y, Z):
    return G.slab(R.prof("hg_lower")(Y, Z), X, HW["hg_lower"], 12.0)


def hg_lower(X, Y, Z):
    return np.maximum(_hgl_outer(X, Y, Z), Y - IY["hg_cap_rear"])


def hg_lower_cavity(X, Y, Z):
    cav = np.maximum.reduce([_hgl_outer(X, Y, Z) + POLY_T, Z - 9.0,                         # true wall offset under a top skin
                             G.yband(Y, IY["hg_cap_rear"] - 4, IY["trunnion_front"] + 4)])
    return np.minimum(cav, np.maximum(np.hypot(X, Z) - 11.5, G.yband(Y, IY["hg_cap_rear"], IY["trunnion_front"] - 2)))  # barrel channel


def hg_cap(X, Y, Z):
    d = np.maximum(_hgl_outer(X, Y, Z) - 0.6, G.yband(Y, IY["hg_cap_front"], IY["hg_cap_rear"]))
    return S.diff(d, S.cyl(X, Y, Z, "y", (0, 0), 7.5, IY["hg_cap_rear"] - 2, IY["hg_cap_front"] + 2))


PARTS = [
    dict(name="gastube", slot="upper_handguard", mat="park", h=0.3, outer=gastube,
         box=(IY["rsight_front"] - 6, IY["gastube_front"] + 2, IZ["piston"] - 12, IZ["piston"] + 12)),
    dict(name="hg_upper", slot="upper_handguard", mat="poly", h=0.4, outer=hg_upper, trace="hg_upper"),
    dict(name="hg_lower", slot="lower_handguard", mat="poly", h=0.35, outer=hg_lower, inner=hg_lower_cavity, trace="hg_lower"),
    dict(name="hg_cap", slot="lower_handguard", mat="park", h=0.3, outer=hg_cap, trace="hg_lower"),
]
