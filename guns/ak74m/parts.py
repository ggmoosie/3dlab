"""parts.py - builds the AK-74M high-poly from the module registry -> work/high.npz + review/modules.json.
Each mod_*.py lists PARTS rows: name, slot (iface.SLOTS), material, voxel h, outer field, optional inner cavity, bbox
(traced part name or explicit (y0, y1, z0, z1)). Surface features (details.FEATURES) go on the OUTER field before the
cavity is cut, so a pocket never punches through a 1 mm wall. Adding a module = adding a mod_*.py to MODULES.
usage: .venv/bin/python guns/ak74m/parts.py [parts=a,b] [h_scale=1.0] [out=path]"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import sdf as S
import ak_ref as R
import ak_geo as G
import details as DT
import iface as I
import mod_front, mod_handguard, mod_receiver, mod_bcg, mod_fcg, mod_furniture

MODULES = (mod_front, mod_handguard, mod_receiver, mod_bcg, mod_fcg, mod_furniture)
ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
ONLY = ARGS["parts"].split(",") if "parts" in ARGS else None
HS = float(ARGS.get("h_scale", 1.0))


def registry():
    rows = [p for m in MODULES for p in m.PARTS]
    names = [p["name"] for p in rows]
    assert len(names) == len(set(names)), "duplicate part names"
    assert all(p["slot"] in I.SLOTS for p in rows), [p["slot"] for p in rows if p["slot"] not in I.SLOTS]
    return {p["name"]: p for p in rows}


def field(p):
    outer, inner, n = p["outer"], p.get("inner"), p["name"]

    def f(X, Y, Z):
        d = DT.apply(n, outer(X, Y, Z), X, Y, Z)
        return np.maximum(d, -inner(X, Y, Z)) if inner else d
    return f


def bounds(p):
    y0, y1, z0, z1 = R.extent(p["trace"]) if "trace" in p else p["box"]
    xh = p.get("xh", 28.0)
    return (-xh, min(y0, y1) - 4, min(z0, z1) - 4), (xh, max(y0, y1) + 4, max(z0, z1) + 4)


def manifest(reg):
    return dict(frame="B mm: bore z=0, muzzle face y=0, gun y<0, +x left", slots=I.SLOTS,
                sockets={k: dict(pos=[round(float(c), 2) for c in v[0]], axis=v[1]) for k, v in I.SOCKETS.items()},
                parts={n: dict(slot=p["slot"], mat=p["mat"], hollow=bool(p.get("inner"))) for n, p in reg.items()})


def main():
    out = ARGS.get("out", os.path.join(HERE, "work", "high.npz"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    reg = registry()
    old = dict(np.load(out)) if (ONLY and os.path.exists(out)) else {}
    parts = {}
    for name, p in reg.items():
        if ONLY and name not in ONLY:
            continue
        lo, hi = bounds(p); h = p["h"] * HS; off = h * 0.3713
        V, F = S.mesh(field(p), np.add(lo, -off), np.add(hi, off), h, lip=3.5, name=name)
        parts[name] = (V, F, {"mat": np.full(len(V), G.MAT[p["mat"]])})
    S.save(out + ".tmp.npz", parts)
    new = dict(np.load(out + ".tmp.npz"))
    for k in old:
        if k.split("__")[0] not in parts and k.split("__")[0] in reg:                    # keep built parts that still exist
            new[k] = old[k]
    np.savez_compressed(out, **new)
    os.remove(out + ".tmp.npz")
    json.dump(manifest(reg), open(os.path.join(HERE, "review", "modules.json"), "w"), indent=1, default=str)
    print("HIGH_DONE", sorted({k.split("__")[0] for k in new}))


if __name__ == "__main__":
    main()
