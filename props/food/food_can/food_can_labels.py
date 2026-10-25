"""Food can variants (invented brands): paper wrap 23.94 × 9.5 cm + the contents surface (Ø7.48 cm) in one print sheet.
x = 0.75 W faces -Y. The food art draws its chunks exactly where the shared relief has bumps (food_can_tex.CHUNKS)."""
import math, os, sys, zlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab.labels import Label, age
from lab import tex
import food_can_geo as G
from food_can_tex import CHUNKS, LABEL_Z

W, H = 2 * math.pi * 3.81, (LABEL_Z[1] - LABEL_Z[0]) / 10
PPC = 2048 / W
FRONT, BACK = 0.75 * W, 0.25 * W
FD = 7.48


def finish(lacquer=(0.78, 0.62, 0.32), label_rough=0.7):
    return dict(ColorA=(0.5, 0.5, 0.5), RoughA=0.4, MetalA=0.0, ColorB=lacquer, RoughB=0.3, MetalB=0.85, LabelRough=label_rough)


def food(sauce, chunk, gloss_spots="#ffffff", extra=None):
    F = Label(FD, FD, 200, bg=sauce)
    rng = np.random.default_rng(2)
    for cx, cy, a, rx, ry in CHUNKS:
        x, y = FD / 2 + cx / 10, FD / 2 - cy / 10
        c = chunk if isinstance(chunk, str) else chunk[int(rng.integers(len(chunk)))]
        F.poly([(x + rx / 10 * math.cos(t) * math.cos(a) - ry / 10 * math.sin(t) * math.sin(a),
                 y - (rx / 10 * math.cos(t) * math.sin(a) + ry / 10 * math.sin(t) * math.cos(a))) for t in np.linspace(0, 2 * math.pi, 20)], c)
    for _ in range(60):
        x, y = rng.uniform(0.3, FD - 0.3, 2)
        F.ellipse(x, y, 0.04, 0.025, (1, 1, 1, 0.5))
    if extra:
        extra(F)
    return F


def panel(L, cx, title, kcal, company, ingredients, ink, bg="#ffffff"):
    w = 5.2
    L.rect(cx - w / 2 - 0.2, 0.3, cx + w / 2 + 0.2, H - 0.3, bg, radius=0.12)
    y = L.nutrition(cx - w / 2, 0.45, w, [("Calories", str(kcal), ""), ("Total Fat", "4g", "5%"), ("Sodium", "780mg", "34%"),
                                           ("Total Carb.", "29g", "11%"), (" Dietary Fiber", "6g", "21%"), ("Protein", "9g", "")],
                    serving=f"Serving size 1/2 cup (130g) · about 3.5 per container", color=ink, bg=bg)
    y = L.para(cx - w / 2, y + 0.12, w, "INGREDIENTS: " + ingredients, "sans_r", 0.15, ink, 1.15)
    L.para(cx - w / 2, y + 0.06, w, company, "sans_r", 0.14, ink, 1.15)
    L.barcode(cx + w / 2 + 0.6, H - 2.3, 2.6, 1.3, "0" + str(zlib.crc32(title.encode()) % 10 ** 11).zfill(11))


def bayside():
    navy, orange = "#16264a", "#e8742a"
    L = Label(W, H, PPC, bg="#f4ead6")
    L.rect(0, 0, W, 1.2, navy); L.rect(0, H - 1.0, W, H, navy)
    L.text(W / 2, 0.6, "BAYSIDE  ·  SINCE 1922  ·  " * 6, "sans", 0.3, "#f4ead6", "mm", tracking=0.15)
    L.ellipse(FRONT, 4.6, 3.0, 2.3, orange)
    for k in range(26):
        a = k * 2.4
        L.ellipse(FRONT + 1.9 * math.cos(a) * (k / 26) ** 0.5, 4.9 + 1.4 * math.sin(a) * (k / 26) ** 0.5, 0.32, 0.22, "#c9572a", outline="#7a2e10", width=0.02)
    L.text(FRONT, 2.0, "BAYSIDE", "display", 1.1, navy, "mm", tracking=0.08)
    L.text(FRONT, 7.4, "BAKED BEANS", "heavy", 0.48, navy, "mm", tracking=0.06)
    L.text(FRONT, 8.1, "in rich tomato sauce", "serif_i", 0.32, orange, "mm")
    L.text(FRONT, H - 0.5, "NET WT 15 OZ (425g)", "sans", 0.26, "#f4ead6", "mm", tracking=0.1)
    panel(L, BACK, "bayside", 140, "Bayside Canning Co., Gloucester, MA 01930", "PREPARED WHITE BEANS, WATER, TOMATO PUREE, SUGAR, SALT, MODIFIED CORN STARCH, ONION POWDER, SPICES.", navy, "#fffaf0")
    return L, food("#9a3a16", ["#b8662e", "#a85a26", "#c4743a"]), finish()


def lone_star():
    red, yellow = "#9e1b1b", "#f2b81e"
    L = Label(W, H, PPC, bg=red)
    L.stripes(0, 0, W, H, (0.55, 0.08, 0.08), 0.6, 0.12)
    star = [(FRONT + 2.2 * math.cos(math.pi / 2 + k * math.pi / 5) * (1 if k % 2 == 0 else 0.42),
             4.3 - 2.2 * math.sin(math.pi / 2 + k * math.pi / 5) * (1 if k % 2 == 0 else 0.42)) for k in range(10)]
    L.poly(star, yellow)
    L.text(FRONT, 1.25, "LONE STAR", "stencil", 0.95, yellow, "mm", tracking=0.06)
    L.text(FRONT, 7.3, "CHILI CON CARNE", "heavy", 0.42, "#ffffff", "mm")
    L.text(FRONT, 8.05, "WITH BEANS  ·  HOT", "sans", 0.3, yellow, "mm", tracking=0.2)
    panel(L, BACK, "lonestar", 260, "Lone Star Provisions, Amarillo, TX 79101", "WATER, BEEF, BEANS, TOMATO PASTE, CHILI PEPPER, ONIONS, CUMIN, SALT, GARLIC, PAPRIKA.", "#3a0a0a")
    return L, food("#5e1a10", ["#7a3a1c", "#4a1f12", "#8c4a26"]), finish()


def mama_rosa():
    green, cream = "#1f5c3a", "#f3ecd8"
    L = Label(W, H, PPC, bg=cream)
    L.rect(0, 0, W, 4.3, green)
    L.rect(0, 4.3, W, 4.45, "#c9a54a")
    L.text(FRONT, 1.5, "Mama Rosa's", "serif_i", 1.0, cream, "mm")
    L.text(FRONT, 3.1, "CONDENSED", "sans", 0.3, "#c9a54a", "mm", tracking=0.35)
    L.text(FRONT, 5.9, "CHICKEN", "serif_b", 0.85, green, "mm", tracking=0.06)
    L.text(FRONT, 7.1, "NOODLE SOUP", "sans", 0.42, green, "mm", tracking=0.18)
    L.ellipse(FRONT, 8.4, 0.6, 0.6, "#c9a54a"); L.text(FRONT, 8.4, "★", "sans", 0.5, cream, "mm")
    panel(L, BACK, "mamarosa", 60, "Mama Rosa's Kitchen, Cranston, RI 02910", "CHICKEN STOCK, ENRICHED EGG NOODLES, CHICKEN, CARROTS, CELERY, SALT, CHICKEN FAT, ONION, PARSLEY.", green, "#fffdf5")
    return L, food("#d8a43a", ["#efe0b0", "#f2e8c8", "#d97a2a"]), finish((0.85, 0.83, 0.78), 0.65)


def sunvale():
    L = Label(W, H, PPC)
    L.gradient(0, 0, W, H, "#ffcf4a", "#ff7b2e")
    for k, (x, y, r) in enumerate([(FRONT - 1.4, 5.2, 1.5), (FRONT + 1.3, 5.6, 1.3), (FRONT, 4.0, 1.2)]):
        L.ellipse(x, y, r, r, "#ff9a3c"); L.ellipse(x - r * 0.25, y - r * 0.25, r * 0.55, r * 0.5, "#ffc06a")
        L.ellipse(x + r * 0.15, y + r * 0.1, r * 0.25, r * 0.3, "#b84a20")
    L.text(FRONT, 1.3, "Sunvale", "script", 1.0, "#ffffff", "mm", stroke=0.04, stroke_color="#b84a20")
    L.text(FRONT, 7.6, "YELLOW CLING PEACH HALVES", "sans", 0.3, "#5a1e06", "mm", tracking=0.08)
    L.text(FRONT, 8.25, "in light syrup", "serif_i", 0.3, "#ffffff", "mm")
    panel(L, BACK, "sunvale", 90, "Sunvale Orchards, Modesto, CA 95354", "PEACHES, WATER, SUGAR, CITRIC ACID.", "#5a1e06", "#fff6e6")
    return L, food("#f0a63a", ["#f6b04a", "#ffbd55", "#eda040"]), finish()


def nonna():
    L = Label(W, H, PPC, bg="#fbf6ec")
    for i, c in enumerate(("#2a7a3a", "#fbf6ec", "#b8262b")):
        L.rect(i * W / 3, 0, (i + 1) * W / 3, 0.6, c)
    L.ellipse(FRONT, 4.4, 2.5, 2.5, "#b8262b")
    for k in range(7):
        a = k * 2 * math.pi / 7
        L.ellipse(FRONT + 1.4 * math.cos(a), 4.4 + 1.4 * math.sin(a), 0.55, 0.55, "#6e3a20", outline="#3a1a0a", width=0.03)
    L.text(FRONT, 1.5, "Nonna's", "serif_i", 0.95, "#b8262b", "mm")
    L.text(FRONT, 7.6, "MEATBALLS", "heavy", 0.5, "#2a2a2a", "mm", tracking=0.06)
    L.text(FRONT, 8.35, "in slow-simmered marinara", "serif_i", 0.3, "#2a7a3a", "mm")
    panel(L, BACK, "nonna", 310, "Nonna's Table Foods, Arthur Ave, Bronx, NY 10458", "MEATBALLS (BEEF, PORK, BREADCRUMBS, EGG, PARMESAN), TOMATO PUREE, WATER, OLIVE OIL, GARLIC, BASIL, SALT.", "#2a2a2a")
    return L, food("#a3241a", ["#5c2e18", "#6e3a20", "#4a2412"]), finish()


def good_boy():
    L = Label(W, H, PPC, bg="#2f6fb5")
    L.halftone(0, 0, W, H, "#4a88cc", 0.35, 0.12, fade="y")
    dx = FRONT                                                 # cartoon dog head
    L.ellipse(dx, 4.5, 1.9, 1.7, "#f2c48a"); L.ellipse(dx - 1.7, 3.6, 0.7, 1.3, "#8a5a2a"); L.ellipse(dx + 1.7, 3.6, 0.7, 1.3, "#8a5a2a")
    L.ellipse(dx - 0.65, 4.1, 0.22, 0.26, "#1a1a1a"); L.ellipse(dx + 0.65, 4.1, 0.22, 0.26, "#1a1a1a")
    L.ellipse(dx, 5.0, 0.45, 0.3, "#1a1a1a"); L.ellipse(dx + 0.2, 5.8, 0.3, 0.45, "#e05a6a")
    L.text(FRONT, 1.3, "GOOD BOY", "heavy", 0.75, "#ffd84a", "mm", stroke=0.04, stroke_color="#1a3a6a")
    L.text(FRONT, 7.5, "CHICKEN & RICE DINNER", "sans", 0.36, "#ffffff", "mm", tracking=0.08)
    L.text(FRONT, 8.25, "COMPLETE NUTRITION FOR ADULT DOGS", "sans", 0.22, "#ffd84a", "mm", tracking=0.1)
    panel(L, BACK, "goodboy", 380, "Good Boy Pet Foods, Paramus, NJ 07652. FOR DOGS ONLY.", "WATER, CHICKEN, LIVER, RICE, MEAT BY-PRODUCTS, GUAR GUM, VITAMINS AND MINERALS.", "#1a3a6a")
    return L, food("#6b4a2e", ["#8a6440", "#5a3a22", "#a07a52"]), finish((0.80, 0.80, 0.80), 0.7)


def mystery():
    """Label long gone: bare tin, marker scrawl, rust bloom. Contents unknown."""
    L = Label(W, H, PPC)
    L.text(FRONT, 3.3, "???", "hand", 1.6, "#141414", "mm", rotate=-6)
    L.text(FRONT - 0.3, 5.4, "beans? meat?", "hand", 0.7, "#141414", "mm", rotate=-6)
    L.text(BACK, 4.6, "DON'T EAT IF BULGING", "hand", 0.55, "#7a1010", "mm", rotate=3)
    a = L.array()
    h, w = a.shape[:2]
    stain = np.clip((tex.fbm(h, w, 5, 9) - 0.62) * 4, 0, 1) * np.clip((np.linspace(1, 0, h)[:, None] - 0.3) * 2, 0, 1)
    rust = np.stack([np.full((h, w), 0.32), np.full((h, w), 0.15), np.full((h, w), 0.06), stain * 0.85], -1).astype(np.float32)
    out = rust.copy()
    out[..., :3] = np.where(a[..., 3:] > 0, a[..., :3], rust[..., :3])
    out[..., 3] = np.maximum(a[..., 3], stain * 0.85)
    L.im = Image.fromarray((out * 255).astype(np.uint8)).resize(L.im.size)
    return L, food("#6a6052", ["#7a705e", "#5a5246", "#8a7f6a"]), finish((0.65, 0.6, 0.5), 0.8)


VARIANTS = {"BaysideBeans": bayside, "LoneStarChili": lone_star, "MamaRosaSoup": mama_rosa, "SunvalePeaches": sunvale,
            "NonnaMeatballs": nonna, "GoodBoyDogFood": good_boy, "Mystery": mystery}


def to_uv(L, F, seed):
    full = np.zeros((2048, 2048, 4), np.float32)
    acc = [0.0]
    for p, q in zip(G.SIDE, G.SIDE[1:]):
        acc.append(acc[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
    zs = [p[1] for p in G.SIDE]
    u0, v0, u1, v1 = G.UV_SIDE
    vt = v0 + (v1 - v0) * np.interp(LABEL_Z[1], zs, acc) / acc[-1]
    vb = v0 + (v1 - v0) * np.interp(LABEL_Z[0], zs, acc) / acc[-1]
    i0, i1 = int(round((1 - vt) * 2048)), int(round((1 - vb) * 2048))
    art = age(L.array(), seed, fade=0.08, scuff=0.12)
    full[i0:i1] = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((2048, i1 - i0), Image.LANCZOS), np.float32) / 255
    u0, v0, u1, v1 = G.UV_FOOD
    j0, j1, k0, k1 = int(round(u0 * 2048)), int(round(u1 * 2048)), int(round((1 - v1) * 2048)), int(round((1 - v0) * 2048))
    full[k0:k1, j0:j1] = np.asarray(F.image().resize((j1 - j0, k1 - k0), Image.LANCZOS), np.float32) / 255
    return full
