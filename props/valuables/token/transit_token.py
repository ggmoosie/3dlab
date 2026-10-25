"""SM_Token: transit-token collection (8 + secret). One Ø23 mm coin mesh; each variant overrides Normal (its relief),
sets metals (ring = layer A, centre = layer B, so bi-metal tokens work) and prints patina into the recesses.
    .venv/bin/python props/valuables/token/transit_token.py [--fast]"""
import os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../..")); sys.path.insert(0, HERE)
import bpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from lab import scene, master, export, tex, geo

FAST = "--fast" in sys.argv
S = 1024
MM = 0.001
R, T = 11.5, 1.7
TEXD, FBXD, REND = (os.path.join(HERE, d) for d in ("textures", "fbx", "renders"))
UV_TOP, UV_BOT, UV_EDGE = (0.0, 0.5, 0.49, 0.99), (0.51, 0.5, 1.0, 0.99), (0.0, 0.0, 1.0, 0.12)
FONT = os.path.join(HERE, "../../../fonts/WorkSans-Bold.ttf")
SERIF = os.path.join(HERE, "../../../fonts/YoungSerif-Regular.ttf")


def coin():
    top = [(0, T - 0.25), (R - 1.3, T - 0.25), (R - 1.0, T), (R - 0.25, T), (R, T - 0.2)]
    edge = [(R, T - 0.2), (R, 0.2)]
    bot = [(R, 0.2), (R - 0.25, 0.0), (R - 1.0, 0.0), (R - 1.3, 0.25), (0, 0.25)]
    sc = lambda pts: [(r * MM, z * MM) for r, z in pts]
    o = geo.lathe("SM_Token", [dict(pts=sc(top[::-1][::-1]), uv=UV_TOP, mode="disc"), dict(pts=sc(edge), uv=UV_EDGE),
                               dict(pts=sc(bot), uv=UV_BOT, mode="disc", flip=True)], segs=48)
    return o


def relief(motif, ring_text, back_text, seed):
    """Height map (0..1) for both faces in UV space: ring lettering, centre motif, back legend; edge reeding."""
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    for rect, text, mo in ((UV_TOP, ring_text, motif), (UV_BOT, back_text, None)):
        u0, v0, u1, v1 = rect
        cx, cy, rad = (u0 + u1) / 2 * S, (1 - (v0 + v1) / 2) * S, (u1 - u0) / 2 * S
        d.ellipse([cx - rad * (R - 1.0) / R, cy - rad * (R - 1.0) / R, cx + rad * (R - 1.0) / R, cy + rad * (R - 1.0) / R], outline=200, width=3)
        f = ImageFont.truetype(FONT, int(rad * 0.13))
        for k, ch in enumerate(text):                              # lettering round the ring
            a = math.pi / 2 + 1.2 - k * (2 * math.pi * 0.86) / max(len(text), 1)
            g = Image.new("L", (80, 80), 0)
            ImageDraw.Draw(g).text((40, 40), ch, font=f, fill=255, anchor="mm")
            g = g.rotate(math.degrees(a - math.pi / 2), resample=Image.BICUBIC)
            x, y = cx + rad * 0.72 * math.cos(a), cy - rad * 0.72 * math.sin(a)
            im.paste(255, (int(x - 40), int(y - 40)), g)
        if mo:
            mo(d, cx, cy, rad)
        else:
            d.text((cx, cy), "GOOD FOR\nONE FARE", font=ImageFont.truetype(FONT, int(rad * 0.16)), fill=255, anchor="mm", align="center")
    h = np.asarray(im, np.float32) / 255
    u, v = tex.uv_grid(S)
    h = np.where(tex.in_rect(u, v, UV_EDGE), (np.abs(np.sin(u * math.pi * 140)) ** 0.5), h)          # reeded edge
    h += tex.fbm(S, S, 30, seed) * 0.06                                                           # wear pitting
    return tex.blur(h, 1.0)


def m_text(t, size=0.42, font=FONT):
    def f(d, cx, cy, rad):
        d.text((cx, cy), t, font=ImageFont.truetype(font, int(rad * size)), fill=255, anchor="mm")
    return f


def m_y(d, cx, cy, rad):                                            # the cut-out "Y" look, as relief
    w = rad * 0.12
    d.line([(cx - rad * 0.3, cy - rad * 0.32), (cx, cy), (cx + rad * 0.3, cy - rad * 0.32)], fill=255, width=int(w))
    d.line([(cx, cy), (cx, cy + rad * 0.36)], fill=255, width=int(w))


def m_bull(d, cx, cy, rad):
    for k, rr in enumerate((0.45, 0.32, 0.18)):
        d.ellipse([cx - rad * rr, cy - rad * rr, cx + rad * rr, cy + rad * rr], outline=255, width=int(rad * 0.05))


def m_star(d, cx, cy, rad):
    pts = [(cx + rad * 0.42 * (1 if k % 2 == 0 else 0.42) * math.cos(-math.pi / 2 + k * math.pi / 5),
            cy + rad * 0.42 * (1 if k % 2 == 0 else 0.42) * math.sin(-math.pi / 2 + k * math.pi / 5)) for k in range(10)]
    d.polygon(pts, fill=255)


def m_skull(d, cx, cy, rad):
    d.ellipse([cx - rad * 0.33, cy - rad * 0.38, cx + rad * 0.33, cy + rad * 0.2], fill=255)
    d.rectangle([cx - rad * 0.2, cy + rad * 0.1, cx + rad * 0.2, cy + rad * 0.32], fill=255)
    for sx in (-1, 1):
        d.ellipse([cx + sx * rad * 0.15 - rad * 0.09, cy - rad * 0.16, cx + sx * rad * 0.15 + rad * 0.09, cy + rad * 0.02], fill=0)


BRASS, BRONZE, NICKEL, COPPER, GOLD = (0.78, 0.62, 0.32), (0.62, 0.40, 0.22), (0.72, 0.72, 0.70), (0.80, 0.45, 0.30), (0.95, 0.75, 0.30)
RING = "CITY TRANSIT AUTHORITY · "
VARIANTS = {   # key: (motif, ring metal, centre metal, lore, rarity)
    "1953_Y": (m_y, BRASS, BRASS, "The first Y-cut token. Fare: 15¢.", "common"),
    "1970_Large": (m_text("NYC", 0.38, SERIF), BRASS, BRASS, "Thirty cents and a bigger coin.", "common"),
    "1980_Solid": (m_text("1980", 0.34), BRONZE, BRONZE, "Solid brass, no cut-out.", "common"),
    "1986_Bullseye": (m_bull, BRASS, NICKEL, "The bullseye: brass ring, steel heart.", "uncommon"),
    "1995_Bimetal": (m_star, NICKEL, BRASS, "Last of the bi-metals before the cards.", "uncommon"),
    "2003_Last": (m_text("LAST\nRIDE", 0.2), COPPER, COPPER, "Struck for the final day of tokens.", "rare"),
    "Commemorative": (m_text("100", 0.4, SERIF), GOLD, GOLD, "Centennial gold. Someone kept it polished.", "rare"),
    "Misprint": (m_text("YNC", 0.38, SERIF), BRASS, BRASS, "Die struck backwards. Collectors lose sleep over this.", "rare"),
    "Arcade_Secret": (m_skull, (0.25, 0.25, 0.27), (0.25, 0.25, 0.27), "Not a fare token. Opens the back room at Coney.", "legendary"),
}


def main():
    scene.reset()
    o = scene.link(coin(), scene.collection("Game"))
    u, v = tex.uv_grid(S)
    ring = np.zeros((S, S), np.float32)
    for rect in (UV_TOP, UV_BOT):
        x, y = tex.disc_map(u, v, rect, R)
        ring = np.maximum(ring, tex.in_rect(u, v, rect) * (np.hypot(x, y) > R * 0.55))
    ring = np.maximum(ring, tex.in_rect(u, v, UV_EDGE))
    one = np.ones((S, S), np.float32)
    p = lambda k: os.path.join(TEXD, f"T_Token_{k}.png")
    os.makedirs(TEXD, exist_ok=True)
    tex.save(tex.lin_to_srgb(np.zeros((S, S, 3)) + 0.7), p("BaseColor"))
    tex.save(np.stack([one, 0.3 + 0.1 * tex.fbm(S, S, 20, 1), one], -1), p("ORM"))
    tex.save(np.stack([ring, 1 - ring, one, 0 * one], -1), p("Mask"), "RGBA")
    tex.save(tex.height_to_normal(np.zeros((S, S)), 1.0), p("Normal"))
    texs = {k: p(k) for k in ("BaseColor", "Normal", "ORM", "Mask")}
    mats, data = {}, {}
    for i, (k, (motif, ringm, centre, lore, rarity)) in enumerate(VARIANTS.items()):
        h = relief(motif, RING * 2 if len(RING) < 20 else RING, "GOOD FOR ONE FARE · NOT FOR SALE · ", i)
        npth = tex.save(tex.height_to_normal(h * 0.25, S / 23 * 0.6), os.path.join(TEXD, f"T_Token_Normal_{k}.png"))
        cav = np.clip(1 - h * 1.4, 0, 1) * tex.fbm(S, S, 10, i + 3)
        pat = np.concatenate([np.zeros((S, S, 3)) + [0.05, 0.06, 0.04], (cav * 0.55)[..., None]], -1)       # patina in the recesses
        lp = tex.save(pat, os.path.join(TEXD, f"T_Token_Label_{k}.png"), "RGBA")
        prm = dict(ColorA=ringm, RoughA=0.32, MetalA=1.0, ColorB=centre, RoughB=0.3, MetalB=1.0, LabelRough=0.7, WearAmount=0.0, Dirt=0.3)
        m = master.material(f"MI_Token_{k}", dict(texs, Normal=npth), prm, label=lp)
        m.use_fake_user = True
        mats[k] = m
        data[k] = dict(params=prm, normal=os.path.basename(npth), label=os.path.basename(lp), lore=lore, rarity=rarity, number=i + 1)
    o.data.materials.append(mats["1953_Y"])
    tri = scene.tris(o)
    h = scene.link(export.hull(o, "UCX_SM_Token_00", 32), scene.collection("Collision")); h.hide_render = True
    export.fbx([o, h], f"{FBXD}/SM_Token.fbx")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "transit_token.blend"), relative_remap=True)
    dups = []
    for i, k in enumerate(VARIANTS):
        d = o.copy(); d.data = o.data.copy(); d.data.materials[0] = mats[k]
        d.location = ((i % 5 - 2) * 0.028, (i // 5) * -0.03, 0); d.rotation_euler.z = i * 0.7
        bpy.context.scene.collection.objects.link(d)
        dups.append(d)
    o.hide_render = True
    scene.studio(dups)
    scene.frame(dups, yaw=0, pitch=62, lens=85, fill=0.9, name="Tokens", aspect=800 / 1600)
    scene.render(f"{REND}/token_set.png", 1600, 16 if FAST else 96, res_y=800)
    export.manifest(os.path.join(HERE, "manifest.json"), {
        "item": "Token", "set": "Token Box", "category": "valuables", "clip": None, "dims_cm": [2.3, 2.3, 0.17],
        "real_object": "23 mm transit token", "pivot": "centre of the bottom face", "tris": {"SM_Token": tri},
        "textures": {"size": S, "normal": "OpenGL; each variant overrides Normal with its relief", "variants": {k: d["label"] for k, d in data.items()}},
        "variant_params": {k: d["params"] for k, d in data.items()}, "variant_normals": {k: d["normal"] for k, d in data.items()},
        "collection": {k: dict(number=d["number"], rarity=d["rarity"], lore=d["lore"]) for k, d in data.items()}})
    print("TRIS", {"SM_Token": tri})


if __name__ == "__main__":
    main()
