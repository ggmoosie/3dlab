"""bl_bake.py - bake every SCAR 17S game mesh from its own high-poly shells, then compose its texture set.
  bakes (work/bake/<Asm>_*.npy): NORMAL (tangent, OpenGL), AO (64 spp, this assembly's shells only), CURV, ID (material),
         POS (world position of the low, for seamless 3-D noise: camo, stipple, smudges)
  textures (textures/T_SCAR17S_<Asm>_*.png): BaseColor (substrate incl. steel wear), Normal, ORM, Mask
         (R anodised = ColorA, G polymer = ColorB, B print region, A wear) + Label_<Variant> (camo) for the receiver parts
usage: .venv/bin/python guns/scar17s/bl_bake.py [asm=Upper,Lower] [fast]"""
import bpy, os, sys, math, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import bake, tex

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
FAST = "fast" in ARGS
BK, TEXD = os.path.join(HERE, "work", "bake"), os.path.join(HERE, "textures")
RECEIVER = ("Upper", "Lower", "Grip", "VGrip", "Stock", "Mag")             # FDE / Black / Camo variants


def noise3(P, scale, seed=0, octaves=4):
    """fbm value noise at world points P (n, 3) metres; scale = features per metre"""
    out = np.zeros(len(P), np.float32); amp, tot = 1.0, 0.0
    rng = np.random.default_rng(seed)
    perm = rng.permutation(4096)
    for o in range(octaves):
        Q = P * scale * (2 ** o) + rng.uniform(0, 100, 3)
        i = np.floor(Q).astype(np.int64); f = (Q - i).astype(np.float32)
        f = f * f * (3 - 2 * f)
        def h(dx, dy, dz):
            k = perm[(perm[(perm[(i[:, 0] + dx) & 4095] + i[:, 1] + dy) & 4095] + i[:, 2] + dz) & 4095]
            return (k / 4095.0).astype(np.float32)
        x00 = h(0, 0, 0) * (1 - f[:, 0]) + h(1, 0, 0) * f[:, 0]
        x10 = h(0, 1, 0) * (1 - f[:, 0]) + h(1, 1, 0) * f[:, 0]
        x01 = h(0, 0, 1) * (1 - f[:, 0]) + h(1, 0, 1) * f[:, 0]
        x11 = h(0, 1, 1) * (1 - f[:, 0]) + h(1, 1, 1) * f[:, 0]
        y0 = x00 * (1 - f[:, 1]) + x10 * f[:, 1]; y1 = x01 * (1 - f[:, 1]) + x11 * f[:, 1]
        out += amp * (y0 * (1 - f[:, 2]) + y1 * f[:, 2]); tot += amp; amp *= 0.5
    return out / tot


def do_bakes(o, size):
    shells = o["shells"].split(",")
    highs = [bpy.data.objects["HI_" + s] for s in shells]
    keep = set(highs) | {o}
    hidden = []
    for x in bpy.data.objects:
        if x not in keep and not x.hide_render:
            x.hide_render = True; hidden.append(x)
    for h in highs:
        h.hide_render = False
    pv = {}
    for m in bpy.data.materials:
        if m.name.startswith("PV_"):
            pv[m.name] = tuple(m.diffuse_color)
            k = int(m.name[3:]); m.diffuse_color = (0.1 * (k + 1), 0, 0, 1)        # 0 = background
    n = o.name.replace("SM_SCAR17S_", "")
    sz = size // 2 if FAST else size
    res = {}
    res["NORMAL"] = bake.bake(o, highs, "NORMAL", sz, f"{BK}/{n}_normal.png", samples=1, extrusion=0.0009, max_ray=0.0025)
    res["AO"] = bake.bake(o, highs, "AO", sz, f"{BK}/{n}_ao.png", samples=16 if FAST else 64, extrusion=0.0009, max_ray=0.0025)
    res["CURV"] = bake.bake(o, highs, "CURV", sz, f"{BK}/{n}_curv.png", samples=1, extrusion=0.0009, max_ray=0.0025)
    res["ID"] = bake.bake(o, highs, "ID", sz, f"{BK}/{n}_id.png", samples=1, extrusion=0.0009, max_ray=0.0025)
    pos, msk = bake.bake(o, [], "POS", sz, f"{BK}/{n}_pos.png", samples=1)
    res["POS"], res["MASK"] = pos, msk
    for m in bpy.data.materials:
        if m.name in pv:
            m.diffuse_color = pv[m.name]
    for x in hidden:
        x.hide_render = False
    for h in highs:
        h.hide_render = True
    if sz != size:                                         # fast mode: upsample so the texture sizes stay final
        from PIL import Image
        def up(a):
            a = np.asarray(a, np.float32)
            if a.ndim == 2:
                return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), np.float32)
            return np.stack([up(a[..., c]) for c in range(a.shape[-1])], -1)
        res = {k: (up(v) if k != "MASK" else up(v.astype(np.float32)) > 0.5) for k, v in res.items()}
    np.savez_compressed(f"{BK}/{n}.npz", **{k: np.asarray(v, np.float16 if k != "MASK" else bool) for k, v in res.items()})
    return res


def compose(n, B, size):
    raw = np.round(B["ID"][..., 0].astype(np.float32) / 0.1).astype(int) - 1
    from scipy import ndimage
    near = ndimage.distance_transform_edt(raw < 0, return_distances=False, return_indices=True)
    ids = np.clip(raw[near[0], near[1]], 0, 7)                                # background -> nearest island id
    anod, poly, steel, park, rub, brass, mag, cop = (ids == k for k in range(8))
    S = size
    P = B["POS"].reshape(-1, 3).astype(np.float32)
    nz = lambda sc, sd, oc=4: noise3(P, sc, sd, oc).reshape(S, S)
    curv = B["CURV"].astype(np.float32)
    ao = np.clip(B["AO"].astype(np.float32), 0, 1)
    ao = np.where(B["MASK"], ao, 1.0)
    edge = np.clip((curv - 0.53) * 7, 0, 1)
    cav = np.clip((0.47 - curv) * 6, 0, 1)
    smudge = nz(18, 3)
    grain = nz(900, 4, 2)
    # substrate colour / roughness / metal per material
    base = np.zeros((S, S, 3), np.float32); rough = np.zeros((S, S), np.float32); metal = np.zeros((S, S), np.float32)
    def put(m, c, r, mt):
        nonlocal base, rough, metal
        base = np.where(m[..., None], np.array(c, np.float32), base); rough = np.where(m, r, rough); metal = np.where(m, mt, metal)
    put(anod, (0.56, 0.56, 0.57), 0.32, 1.0)
    put(poly, (0.30, 0.25, 0.17), 0.62, 0.0)
    put(steel, (0.032, 0.032, 0.034), 0.36, 0.85)
    put(park, (0.042, 0.042, 0.040), 0.62, 0.55)
    put(rub, (0.022, 0.022, 0.022), 0.86, 0.0)
    put(mag, (0.028, 0.028, 0.030), 0.48, 0.55)
    put(brass, (0.70, 0.46, 0.20), 0.26, 1.0)
    put(cop, (0.74, 0.38, 0.24), 0.30, 1.0)
    steelish = steel | park | mag
    wear_s = np.clip(edge * (0.35 + nz(60, 7)) * 1.4, 0, 1) * steelish
    base = base * (1 - wear_s[..., None]) + np.array([0.22, 0.22, 0.23], np.float32) * wear_s[..., None]
    metal = np.where(steelish, np.maximum(metal, wear_s), metal)
    rough = rough + (smudge - 0.5) * 0.12 + (grain - 0.5) * 0.05 - wear_s * 0.12
    base = base * (1 - 0.35 * cav[..., None] * (anod | poly | rub)[..., None])          # dust in recesses
    rough = np.where(brass | cop, rough - 0.08 * smudge, rough)
    # normal: baked + polymer stipple on the grip, rubber texture on the buttpad, fine cast texture on polymer
    nrm = B["NORMAL"].astype(np.float32)
    H = np.zeros((S, S), np.float32)
    # stipple: a material layer inside the traced panels (the clay only carries the panel recess). Texel world position ->
    # B frame -> relief stipple field (photo-traced) -> fine stipple height, extra cavity, slightly darker
    stip = np.zeros((S, S), np.float32)
    if n in ("Grip", "VGrip", "Lower"):
        import relief, scar_spec as SP
        o = np.array(SP.ORIGIN_B)
        Pw = B["POS"].reshape(-1, 3).astype(np.float64)
        yb, zb = -Pw[:, 1] * 1000 + o[1], Pw[:, 2] * 1000 + o[2]
        f = relief.field("stipple")(yb, zb).reshape(S, S)
        stip = (np.clip(-f / 0.5, 0, 1) * B["MASK"] * (np.abs(Pw[:, 0]).reshape(S, S) > 0.003)).astype(np.float32)
        dots = np.clip((nz(1300, 41, 2) - 0.45) * 3.5, 0, 1) ** 1.5
        H += (dots - 0.4) * 2.2 * stip
        ao = ao * (1 - 0.35 * stip * (1 - dots))
    H += (nz(1800, 12, 2) - 0.5) * 0.25 * (poly | rub)
    H += (nz(3000, 13, 1) - 0.5) * 0.08 * anod
    if H.any():
        det = tex.height_to_normal(tex.blur(H, 0.6), S / 400.0)
        nrm = tex.blend_normals(nrm, det)
    # masks
    wear_a = np.clip(edge * (0.4 + nz(40, 21)) * 1.5 + (nz(9, 22) > 0.72) * nz(250, 23) * 0.6, 0, 1) * anod
    wear_a = np.maximum(wear_a, np.clip(edge * nz(30, 24) * 0.6, 0, 1) * poly)
    mask = np.stack([anod.astype(np.float32), poly.astype(np.float32), (anod | poly).astype(np.float32), wear_a], -1)
    p = lambda k: os.path.join(TEXD, f"T_SCAR17S_{n}_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(np.clip(base, 0, 1)), p("BaseColor"))
    tex.save(np.clip(nrm, 0, 1), p("Normal"))
    tex.save(np.stack([ao * (1 - 0.3 * cav), np.clip(rough, 0.05, 1), np.clip(metal, 0, 1)], -1), p("ORM"))
    tex.save(mask, p("Mask"), "RGBA")
    out = {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
    if n in RECEIVER:                                       # camo print (3-D noise, seamless), alpha 0 off the receiver
        a, b, c = nz(14, 31, 3), nz(22, 32, 3), nz(60, 33, 2)
        cols = np.array([(0.42, 0.33, 0.20), (0.24, 0.22, 0.13), (0.14, 0.11, 0.07), (0.55, 0.47, 0.32)], np.float32)
        idx = np.where(a > 0.58, 2, np.where(b > 0.55, 1, np.where(c > 0.62, 3, 0)))
        rgb = tex.lin_to_srgb(cols[idx])
        cam = np.concatenate([rgb, (anod | poly)[..., None].astype(np.float32)], -1)
        out["Label_Camo"] = tex.save(cam, p("Label_Camo"), "RGBA")
        clear = np.zeros((S, S, 4), np.float32)
        out["Label_None"] = tex.save(clear, p("Label_None"), "RGBA")
    return out


def asm_hash(o, size):
    """fingerprint of everything a bake depends on: the low mesh (positions, UVs, faces), every high shell's geometry +
    materials, the texture size and the bake settings -> unchanged assemblies reuse their cached bakes"""
    import hashlib
    hs = hashlib.sha1()

    def mesh(me):
        v = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", v); hs.update(np.round(v, 6).tobytes())
        f = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("material_index", f); hs.update(f.tobytes())
        lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv); hs.update(lv.tobytes())
        if me.uv_layers:
            uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers.active.data.foreach_get("uv", uv); hs.update(np.round(uv, 6).tobytes())
    mesh(o.data)
    for sh in o["shells"].split(","):
        hi = bpy.data.objects["HI_" + sh]; mesh(hi.data); hs.update(np.array(hi.matrix_world, np.float32).tobytes())
    hs.update(f"{size}|{FAST}|v1".encode())
    return hs.hexdigest()


def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "work", "scar_low.blend"))
    os.makedirs(BK, exist_ok=True)
    only = ARGS.get("asm", "").split(",") if ARGS.get("asm") else None
    report = {}
    for o in [x for x in bpy.data.objects if x.name.startswith("SM_SCAR17S_")]:
        n = o.name.replace("SM_SCAR17S_", "")
        if only and n not in only:
            continue
        size = int(o["tex_size"])
        print("[bake]", n, size, flush=True)
        h = asm_hash(o, size)
        cache, hp = os.path.join(BK, f"{n}.npz"), os.path.join(BK, f"{n}.hash")
        if os.path.exists(cache) and os.path.exists(hp) and open(hp).read() == h:
            print("[bake] unchanged -> cached bakes", n, flush=True)
            B = dict(np.load(cache))
        else:
            B = do_bakes(o, size)
            open(hp, "w").write(h)
        report[n] = compose(n, B, size)
        print("[bake] done", n, flush=True)
    json.dump(report, open(os.path.join(HERE, "work", "textures.json"), "w"), indent=1)
    print("BAKE_DONE")


if __name__ == "__main__":
    main()
