"""bl_bake.py - bake every AK-74M game mesh from its own high-poly shells, then compose its texture set.
  bakes (work/bake/<Asm>.npz): NORMAL (tangent, OpenGL), AO, CURV, ID (material), POS (world position of the low)
  textures (textures/T_AK74_<Asm>_*.png): BaseColor (substrate = bare steel / raw polymer, incl. edge wear), Normal, ORM,
  Mask (universal mask: R steel finish A, G polymer B, B print area, A wear) + Label_<Variant> (print: sight range numbers in
  white paint, the generic serial stamped on the left lip). Grip checkering, barrel / brake grain and the stamped serial
  are material detail (normal + AO), the clay keeps the shape.
usage: .venv/bin/python guns/ak74/bl_bake.py [asm=Receiver,Grip] [fast]"""
import bpy, os, sys, math, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import bake, tex
import ak_spec as P
import ak_ref as R

ARGS = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:])
FAST = "fast" in ARGS
BK, TEXD = os.path.join(HERE, "work", "bake"), os.path.join(HERE, "textures")
SERIAL = "93 1047265"                      # generic serial (no factory proof mark, no maker's logo)
FONT = os.path.join(HERE, "../../fonts/WorkSans-Bold.ttf")


def noise3(P_, scale, seed=0, octaves=4):
    out = np.zeros(len(P_), np.float32); amp, tot = 1.0, 0.0
    rng = np.random.default_rng(seed); perm = rng.permutation(4096)
    for o in range(octaves):
        Q = P_ * scale * (2 ** o) + rng.uniform(0, 100, 3)
        i = np.floor(Q).astype(np.int64); f = (Q - i).astype(np.float32); f = f * f * (3 - 2 * f)
        def h(dx, dy, dz):
            k = perm[(perm[(perm[(i[:, 0] + dx) & 4095] + i[:, 1] + dy) & 4095] + i[:, 2] + dz) & 4095]
            return (k / 4095.0).astype(np.float32)
        x00 = h(0, 0, 0) * (1 - f[:, 0]) + h(1, 0, 0) * f[:, 0]; x10 = h(0, 1, 0) * (1 - f[:, 0]) + h(1, 1, 0) * f[:, 0]
        x01 = h(0, 0, 1) * (1 - f[:, 0]) + h(1, 0, 1) * f[:, 0]; x11 = h(0, 1, 1) * (1 - f[:, 0]) + h(1, 1, 1) * f[:, 0]
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
            k = int(m.name[3:]); m.diffuse_color = (0.1 * (k + 1), 0, 0, 1)
    n = o.name.replace("SM_AK74_", "")
    sz = size // 2 if FAST else size
    kw = dict(extrusion=0.0008, max_ray=0.0022)
    res = {}
    res["NORMAL"] = bake.bake(o, highs, "NORMAL", sz, f"{BK}/{n}_normal.png", samples=1, **kw)
    res["AO"] = bake.bake(o, highs, "AO", sz, f"{BK}/{n}_ao.png", samples=16 if FAST else 48, **kw)
    res["CURV"] = bake.bake(o, highs, "CURV", sz, f"{BK}/{n}_curv.png", samples=1, **kw)
    res["ID"] = bake.bake(o, highs, "ID", sz, f"{BK}/{n}_id.png", samples=1, **kw)
    pos, msk = bake.bake(o, [], "POS", sz, f"{BK}/{n}_pos.png", samples=1)
    res["POS"], res["MASK"] = pos, msk
    for m in bpy.data.materials:
        if m.name in pv:
            m.diffuse_color = pv[m.name]
    for x in hidden:
        x.hide_render = False
    for h in highs:
        h.hide_render = True
    if sz != size:
        from PIL import Image
        def up(a):
            a = np.asarray(a, np.float32)
            if a.ndim == 2:
                return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), np.float32)
            return np.stack([up(a[..., c]) for c in range(a.shape[-1])], -1)
        res = {k: (up(v) if k != "MASK" else up(v.astype(np.float32)) > 0.5) for k, v in res.items()}
    np.savez_compressed(f"{BK}/{n}.npz", **{k: np.asarray(v, np.float16 if k != "MASK" else bool) for k, v in res.items()})
    return res


def text_raster(text, h_mm, res=0.05, spacing=0.12):
    """white text on black, cell height h_mm, returned with its mm size (for sampling by B-frame coordinates)"""
    from PIL import Image, ImageDraw, ImageFont
    px = int(h_mm / res * 1.35)
    f = ImageFont.truetype(FONT, px)
    w = int(sum(f.getlength(c) for c in text) + spacing * px * len(text)) + 4
    im = Image.new("L", (w, int(px * 1.3)), 0); d = ImageDraw.Draw(im); x = 2
    for c in text:
        d.text((x, 0), c, font=f, fill=255); x += f.getlength(c) + spacing * px
    a = np.asarray(im, np.float32) / 255
    rows = np.nonzero(a.max(1) > 0.1)[0]; a = a[rows.min():rows.max() + 1]
    return a, a.shape[1] * res * (h_mm / (a.shape[0] * res)), h_mm


def sample_raster(a, wmm, hmm, u, v):
    """u in [0, wmm) left->right, v in [0, hmm) top->bottom -> value (0 outside)"""
    H, W = a.shape
    iu = np.clip((u / wmm * W).astype(int), 0, W - 1); iv = np.clip((v / hmm * H).astype(int), 0, H - 1)
    inside = (u >= 0) & (u < wmm) & (v >= 0) & (v < hmm)
    return np.where(inside, a[iv, iu], 0.0)


def prints(n, B, Pb, S_):
    """print layer: (alpha, colour, stamp height) per texel. Pb = texel positions in the B frame (mm)"""
    x, y, z = Pb[:, 0], Pb[:, 1], Pb[:, 2]
    alpha = np.zeros(len(x), np.float32); col = np.ones((len(x), 3), np.float32); stamp = np.zeros(len(x), np.float32)
    region = np.zeros(len(x), bool)
    if n == "Receiver":
        # serial: left side lip just behind the rear sight block (163320), 3.6 mm digits, read from the left (rear = right)
        a, wmm, hmm = text_raster(SERIAL, 3.6)
        y0 = R.bp(690.0, 266.0)[0]
        zc = R.bp(640.0, 268.6)[1]
        on = (x > P.HW_REC - 0.8) & (x < P.HW_REC + 0.9)
        t = sample_raster(a, wmm, hmm, y0 - y, (zc + hmm / 2) - z) * on
        stamp += t
        region |= on & (y < y0 + 2) & (y > y0 - wmm - 2) & (np.abs(z - zc) < hmm)
        alpha += t * 0.85; col[t > 0] = (0.16, 0.16, 0.17)
        # range numbers on the leaf top (163315): even + battle letter on the left half, odd on the right, rear -> front
        ztop = R.bp(720.0, 233.9)[1]
        leaf = (np.abs(x) < 7.0) & (z > ztop - 0.6) & (y > R.bp(701.0, 234)[0]) & (y < R.bp(757.0, 233)[0])
        region |= leaf
        ys = np.linspace(R.bp(704.0, 234)[0], R.bp(752.0, 233)[0], 6)
        for i, (ev, od) in enumerate(zip(["П", "2", "4", "6", "8", "10"], ["", "1", "3", "5", "7", "9"])):
            for txt, xc in ((ev, 3.4), (od, -3.4)):
                if not txt:
                    continue
                a, wmm, hmm = text_raster(txt, 2.4)
                # read from behind the gun: u along -x (shooter's left -> right), v along -y (front at the top)
                u = (wmm / 2) - (x - xc) * (-1)
                v = (hmm / 2) - (y - ys[i])
                t = sample_raster(a, wmm, hmm, u, v) * leaf
                alpha = np.maximum(alpha, t); col[t > 0.3] = (0.92, 0.92, 0.9)
            tick = leaf & (np.abs(y - ys[i] - 2.2) < 0.18) & (np.abs(np.abs(x) - 1.6) < 1.0)
            alpha = np.maximum(alpha, tick.astype(np.float32)); col[tick] = (0.92, 0.92, 0.9)
    return alpha.reshape(S_, S_), col.reshape(S_, S_, 3), stamp.reshape(S_, S_), region.reshape(S_, S_)


def grip_checker(Pb, hw, S_):
    """diamond checkering inside the grip panels (detail.py panel: (495, 330) -> (477, 405), +-6.8 mm), 1.25 mm pitch"""
    a, b = R.bp(495.0, 330.0), R.bp(477.0, 405.0)
    ang = math.atan2(b[1] - a[1], b[0] - a[0]); L = math.hypot(b[0] - a[0], b[1] - a[1])
    x, y, z = Pb[:, 0], Pb[:, 1], Pb[:, 2]
    u = (y - a[0]) * math.cos(ang) + (z - a[1]) * math.sin(ang)
    v = -(y - a[0]) * math.sin(ang) + (z - a[1]) * math.cos(ang)
    inside = (u > 0.6) & (u < L - 0.6) & (np.abs(v) < 6.2) & (np.abs(x) > hw - 3.6)
    p = 1.25
    s1 = np.abs(((u + v) / p) % 1.0 - 0.5); s2 = np.abs(((u - v) / p) % 1.0 - 0.5)
    hgt = 1.0 - 2.0 * np.maximum(s1, s2)                 # pyramids
    return (hgt * inside).reshape(S_, S_), inside.reshape(S_, S_)


def compose(n, B, size):
    from scipy import ndimage
    raw = np.round(B["ID"][..., 0].astype(np.float32) / 0.1).astype(int) - 1
    near = ndimage.distance_transform_edt(raw < 0, return_distances=False, return_indices=True)
    ids = np.clip(raw[near[0], near[1]], 0, 7)
    park, poly, steel, bright, rub, brass, magpoly, paint = (ids == k for k in range(8))
    S_ = size
    Pw = B["POS"].reshape(-1, 3).astype(np.float64)
    o = np.array(P.ORIGIN_B)
    Pb = np.c_[Pw[:, 0] * 1000 + o[0], -Pw[:, 1] * 1000 + o[1], Pw[:, 2] * 1000 + o[2]]
    P32 = Pw.astype(np.float32)
    nz = lambda sc, sd, oc=4: noise3(P32, sc, sd, oc).reshape(S_, S_)
    curv = B["CURV"].astype(np.float32)
    ao = np.where(B["MASK"], np.clip(B["AO"].astype(np.float32), 0, 1), 1.0)
    edge = np.clip((curv - 0.53) * 7, 0, 1); cav = np.clip((0.47 - curv) * 6, 0, 1)
    smudge, grain = nz(18, 3), nz(900, 4, 2)
    base = np.zeros((S_, S_, 3), np.float32); rough = np.zeros((S_, S_), np.float32); metal = np.zeros((S_, S_), np.float32)

    def put(m, c, r, mt):
        nonlocal base, rough, metal
        base = np.where(m[..., None], np.array(c, np.float32), base); rough = np.where(m, r, rough); metal = np.where(m, mt, metal)
    # substrate = what shows where the finish wears through: bare steel under the black phosphate / bluing, raw polymer
    steelish = park | steel | bright
    put(park, (0.30, 0.30, 0.31), 0.32, 1.0)
    put(steel, (0.33, 0.33, 0.34), 0.28, 1.0)
    put(bright, (0.52, 0.52, 0.53), 0.22, 1.0)               # polished carrier / latch plate
    put(poly | magpoly, (0.055, 0.055, 0.057), 0.55, 0.0)
    put(rub, (0.03, 0.03, 0.03), 0.85, 0.0)
    put(brass, (0.70, 0.46, 0.20), 0.26, 1.0)
    put(paint, (0.8, 0.8, 0.78), 0.6, 0.0)
    rough = rough + (smudge - 0.5) * 0.10 + (grain - 0.5) * 0.05
    base = base * (1 - 0.3 * cav[..., None] * (poly | magpoly)[..., None])
    # normal: baked + material detail
    nrm = B["NORMAL"].astype(np.float32)
    H = np.zeros((S_, S_), np.float32)
    H += (nz(1500, 12, 2) - 0.5) * 0.35 * (poly | magpoly)                      # polymer cast texture
    H += (nz(2600, 13, 1) - 0.5) * 0.22 * (park | steel)                         # phosphate grain
    if n in ("Muzzle", "Barrel"):
        H += (nz(3500, 14, 1) - 0.5) * 0.35 * (park | steel)                     # bead-blasted grain (163337 / 163339)
    if n == "Grip":
        ck, inside = grip_checker(Pb, P.HW_GRIP, S_)
        H += ck * 1.6 * B["MASK"]
        ao = ao * (1 - 0.35 * inside * (1 - ck))
        rough = np.where(inside, rough + 0.06, rough)
    alpha, pcol, stamp, region = prints(n, B, Pb, S_)
    H -= stamp * 1.2                                                            # stamped digits
    ao = ao * (1 - 0.3 * stamp)
    if H.any():
        nrm = tex.blend_normals(nrm, tex.height_to_normal(tex.blur(H, 0.6), S_ / 500.0))
    # wear mask (A): edges and handling areas; the steel finish rubs to bare steel, polymer only polishes a little
    wear = np.clip(edge * (0.35 + nz(45, 21)) * 1.4, 0, 1)
    wear = np.where(poly | magpoly, wear * 0.35, wear)
    mask = np.stack([steelish.astype(np.float32), (poly | magpoly | rub).astype(np.float32), region.astype(np.float32), wear], -1)
    p = lambda k: os.path.join(TEXD, f"T_AK74_{n}_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(np.clip(base, 0, 1)), p("BaseColor"))
    tex.save(np.clip(nrm, 0, 1), p("Normal"))
    tex.save(np.stack([ao * (1 - 0.3 * cav), np.clip(rough, 0.05, 1), np.clip(metal, 0, 1)], -1), p("ORM"))
    tex.save(mask, p("Mask"), "RGBA")
    out = {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
    lab = np.concatenate([tex.lin_to_srgb(pcol), np.clip(alpha, 0, 1)[..., None]], -1)
    out["Label"] = tex.save(lab, p("Label"), "RGBA")
    return out


def main():
    bpy.ops.wm.open_mainfile(filepath=os.path.join(HERE, "work", "ak_low.blend"))
    os.makedirs(BK, exist_ok=True)
    only = ARGS.get("asm", "").split(",") if ARGS.get("asm") else None
    path = os.path.join(HERE, "work", "textures.json")
    report = json.load(open(path)) if os.path.exists(path) else {}
    for o in [x for x in bpy.data.objects if x.name.startswith("SM_AK74_")]:
        n = o.name.replace("SM_AK74_", "")
        if only and n not in only:
            continue
        size = int(o["tex_size"])
        print("[bake]", n, size, flush=True)
        B = do_bakes(o, size)
        report[n] = compose(n, B, size)
        json.dump(report, open(path, "w"), indent=1)
        print("[bake] done", n, flush=True)
    print("BAKE_DONE")


if __name__ == "__main__":
    main()
