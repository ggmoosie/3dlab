"""Variant prints for the 355 mL can (invented brands). Each returns a Label laid out on the unrolled side:
width = circumference (20.8 cm), height = printable arc; x = 0.75 W faces Blender -Y (under the opening, toward
the player when drinking), x = 0.25 W is the back panel. Ink alpha < 1 lets the aluminium show (metallic inks)."""
import math, os, sys, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab.labels import Label, age
from lab import tex
import can_geo as G

ACC = G.side_arc()
TOT = ACC[-1] / 10                          # cm
W = 2 * math.pi * 3.31                      # cm
PPC = 2048 / W                              # px per cm, matches the texture


def zy(z_mm):
    """Label y (cm from the top) of a height on the can."""
    s = np.interp(z_mm, [p[1] for p in G.SIDE], ACC)
    return TOT - s / 10


FRONT, BACK = 0.75 * W, 0.25 * W
ALU = (0.86, 0.86, 0.87)


def finish(tab=ALU, tab_metal=1.0, lid=ALU, lid_metal=1.0, lid_rough=0.30, label_rough=0.22):
    """Layer A = tab, layer B = lid panel, print = side label."""
    return dict(ColorA=tab, RoughA=0.26, MetalA=tab_metal, ColorB=lid, RoughB=lid_rough, MetalB=lid_metal, LabelRough=label_rough)
TOP, BOT = zy(117), zy(4)          # printable band
BODY_T, BODY_B = zy(104), zy(11)


def info_panel(L, cx, kcal, sugar, caf, company, ingredients, ink="#1a1a1a", bg="#ffffff", alc=None):
    w = 5.4
    x0 = cx - w / 2
    y = BODY_T + 0.35
    rows = [("Calories", str(kcal), ""), ("Total Fat", "0g", "0%"), ("Sodium", "45mg", "2%"),
            ("Total Carb.", f"{sugar + 1}g", f"{round((sugar + 1) / 2.75)}%"), (" Total Sugars", f"{sugar}g", ""),
            ("  Incl. Added Sugars", f"{sugar}g", f"{sugar * 2}%"), ("Protein", "0g", "")]
    if alc:
        rows = [("Calories", str(kcal), ""), ("Carbohydrate", "11g", ""), ("Protein", "1g", ""), ("Fat", "0g", "")]
    y = L.nutrition(x0, y, w, rows, serving="Serving size 1 can (355 mL)", color=ink, bg=bg)
    y = L.para(x0, y + 0.18, w, "INGREDIENTS: " + ingredients, "sans_r", 0.17, ink, 1.18)
    if caf:
        y = L.para(x0, y + 0.06, w, f"CAFFEINE CONTENT: {caf} mg / 12 FL OZ", "sans", 0.17, ink)
    L.para(x0, y + 0.06, w, company, "sans_r", 0.16, ink)
    L.barcode(cx - 1.5, BODY_B - 1.75, 3.0, 1.15, digits="0" + str(zlib.crc32(company.encode()) % 10 ** 11).zfill(11), color="#111111")


def bands(L, color, neck_text, bottom_text, ink="#ffffff", neck_color=None):
    L.rect(0, TOP, W, BOT, color)
    L.rect(0, TOP, W, zy(105.5), neck_color or color)
    L.text(W / 2, (TOP + zy(105.5)) / 2, (neck_text + "   ") * 4, "sans", 0.30, ink, "mm", tracking=0.25)
    L.rect(0, BODY_B, W, BOT, neck_color or color)
    L.text(FRONT, (BODY_B + BOT) / 2, bottom_text, "sans", 0.28, ink, "mm", tracking=0.15)
    L.text(BACK, (BODY_B + BOT) / 2, "PLEASE RECYCLE  •  NY VT ME CT MA 5¢  •  MI 10¢", "sans", 0.2, ink, "mm", tracking=0.1)


def kolossal():
    L = Label(W, TOT, PPC)
    red, deep = "#c8102e", "#8e0b20"
    L.gradient(0, TOP, W, BOT, "#d4142f", "#9c0d22")
    bands(L, None, "KOLOSSAL  •  ICE COLD  •  SINCE 1931", "12 FL OZ (355 mL)", neck_color=deep)
    for k in range(240):                                           # the white wave, with a bare-metal pinstripe
        x = W * k / 239
        y = (BODY_T + BODY_B) / 2 + 1.6 * math.sin(2 * math.pi * x / W * 2 - math.pi / 2) + 0.4
        L.ellipse(x, y, 0.20, 0.62, "#ffffff")
    for k in range(480):                                           # bare-metal pinstripe (ink erased)
        x = W * k / 479
        y = (BODY_T + BODY_B) / 2 + 1.6 * math.sin(2 * math.pi * x / W * 2 - math.pi / 2) + 0.4 + 0.95
        L.ellipse(x, y, 0.05, 0.05, (0, 0, 0, 0))
    rng = np.random.default_rng(3)
    for _ in range(70):                                            # rising bubbles around the front
        x, y = FRONT + rng.normal(0, 2.6), rng.uniform(BODY_T + 0.5, BODY_B - 0.5)
        r = rng.uniform(0.04, 0.22)
        L.ellipse(x, y, r, r, None, "#ffffff", 0.025)
    L.text(FRONT, zy(80), "Kolossal", "script", 1.75, "#ffffff", "mm", stroke=0.06, stroke_color=deep, rotate=6)
    L.text(FRONT, zy(61), "C O L A", "heavy", 0.62, "#ffffff", "mm", tracking=0.18)
    L.text(FRONT, zy(55), "ORIGINAL TASTE  •  REAL CANE SUGAR", "sans", 0.2, "#ffd9df", "mm", tracking=0.12)
    L.rect(BACK - 2.9, BODY_T + 0.2, BACK + 2.9, BODY_B - 0.05, "#ffffff", radius=0.15)
    info_panel(L, BACK, 150, 39, 34, "Kolossal Beverage Co., Long Island City, NY 11101  ·  kolossal.nyc",
               "CARBONATED WATER, CANE SUGAR, CARAMEL COLOR, PHOSPHORIC ACID, NATURAL FLAVORS, CAFFEINE.")
    return L, finish(tab=(0.55, 0.05, 0.06), label_rough=0.2)


def zing():
    L = Label(W, TOT, PPC)
    L.gradient(0, TOP, W, BOT, "#c8e63c", "#3f9d2f")
    bands(L, None, "ZING!  •  LEMON-LIME  •  CAFFEINE FREE", "12 FL OZ (355 mL)", ink="#1d4d17", neck_color="#f4f7c8")
    rng = np.random.default_rng(5)
    for k, (x, y, r, c) in enumerate([(FRONT - 3.2, zy(85), 2.3, "#fff36b"), (FRONT + 3.0, zy(35), 2.6, "#9be04a"),
                                     (BACK + 4.6, zy(70), 1.8, "#fff36b"), (2.0, zy(40), 2.0, "#9be04a")]):
        L.ellipse(x, y, r, r, c)                                   # citrus slices
        L.ellipse(x, y, r * 0.88, r * 0.88, "#ffffff")
        for sgm in range(10):
            a0 = 2 * math.pi * sgm / 10
            pts = [(x, y)] + [(x + r * 0.82 * math.cos(a0 + t * 0.55), y + r * 0.82 * math.sin(a0 + t * 0.55)) for t in np.linspace(0.06, 1, 6)]
            L.poly(pts, c)
    L.burst(FRONT + 2.4, zy(97), 0.55, 0.8, 14, "#1d4d17", rot=8)
    L.text(FRONT + 2.4, zy(97) - 0.12, "0g", "sans", 0.22, "#ffffff", "mm")
    L.text(FRONT + 2.4, zy(97) + 0.14, "CAF", "sans", 0.16, "#ffffff", "mm")
    L.text(FRONT, zy(66), "ZING!", "heavy", 1.55, "#ffffff", "mm", stroke=0.09, stroke_color="#1d4d17", rotate=8)
    L.text(FRONT, zy(49), "LEMON-LIME SODA", "sans_i", 0.42, "#1d4d17", "mm", tracking=0.06)
    L.rect(BACK - 2.9, BODY_T + 0.2, BACK + 2.9, BODY_B - 0.05, "#ffffff", radius=0.15)
    info_panel(L, BACK, 140, 37, 0, "Zing Bottling Group, Yonkers, NY 10701", "CARBONATED WATER, HIGH FRUCTOSE CORN SYRUP, "
               "CITRIC ACID, NATURAL LEMON AND LIME FLAVORS, SODIUM CITRATE, SODIUM BENZOATE (PRESERVES FRESHNESS).", ink="#1d4d17")
    return L, finish(tab=(0.80, 0.78, 0.30))


def volt():
    L = Label(W, TOT, PPC)
    L.rect(0, TOP, W, BOT, (0.04, 0.045, 0.05))
    L.stripes(0, TOP, W, BOT, (0.10, 0.11, 0.12), 0.35, 0.12)       # fine matte pinstripes
    bolt = [(0.2, -3.6), (1.4, -3.6), (0.5, -0.6), (1.7, -0.6), (-0.9, 3.9), (-0.1, 0.6), (-1.3, 0.6)]
    for cx, sc, a in ((FRONT + 1.9, 1.25, 0.88), (BACK + 3.9, 0.9, 0.8), (W - 0.6, 0.7, 0.7)):
        L.poly([(cx + x * sc, zy(62) + y * sc) for x, y in bolt], (0.12, 0.85, 1.0, a))
    L.text(FRONT - 0.6, zy(88), "VOLT", "tech", 1.45, "#ffffff", "mm", tracking=0.08, stroke=0.05, stroke_color="#0a0b0c")
    L.text(FRONT - 0.3, zy(35), "ENERGY", "heavy", 0.42, "#ffffff", "mm", tracking=0.35)
    L.text(FRONT - 0.3, zy(28), "160 mg CAFFEINE  •  TAURINE  •  B-VITAMINS", "sans", 0.17, "#9fdcff", "mm", tracking=0.08)
    L.text(W / 2, (TOP + zy(105.5)) / 2, "VOLT  •  STAY CHARGED  •  " * 4, "tech", 0.28, "#22d8ff", "mm", tracking=0.2)
    L.text(FRONT, (BODY_B + BOT) / 2, "12 FL OZ (355 mL)", "sans", 0.28, "#ffffff", "mm", tracking=0.15)
    L.rect(BACK - 2.9, BODY_T + 0.2, BACK + 2.9, BODY_B - 0.05, "#f2f2f2", radius=0.15)
    info_panel(L, BACK, 160, 41, 160, "VOLT Drinks LLC, Newark, NJ 07102. Not recommended for children, pregnant or nursing women.",
               "CARBONATED WATER, SUGAR, CITRIC ACID, TAURINE, SODIUM CITRATE, NATURAL FLAVORS, CAFFEINE, NIACINAMIDE, "
               "PYRIDOXINE HYDROCHLORIDE, CYANOCOBALAMIN, COLOR ADDED.")
    return L, finish(tab=(0.03, 0.03, 0.035), lid=(0.035, 0.035, 0.04), lid_rough=0.45, label_rough=0.55)


def lager():
    L = Label(W, TOT, PPC)
    cream, navy = "#efe6d0", "#1d2b4a"
    L.rect(0, TOP, W, BOT, cream)
    bands(L, None, "FIVE BOROUGHS BREWING  •  QUEENS, NY", "12 FL OZ  •  4.8% ALC./VOL.", neck_color=navy)
    rng = np.random.default_rng(11)
    x, base = 0.0, zy(36)
    while x < W:                                                   # skyline: invented blocks with water towers
        bw, bh = rng.uniform(0.35, 1.1), rng.uniform(0.6, 2.0)
        L.rect(x, base - bh, x + bw, base, navy)
        if rng.random() < 0.35:
            tx = x + bw * 0.5
            L.rect(tx - 0.18, base - bh - 0.32, tx + 0.18, base - bh - 0.08, navy, radius=0.04)
            L.poly([(tx - 0.2, base - bh - 0.32), (tx, base - bh - 0.48), (tx + 0.2, base - bh - 0.32)], navy)
        for wy in np.arange(base - bh + 0.25, base - 0.15, 0.3):
            for wx in np.arange(x + 0.1, x + bw - 0.1, 0.22):
                if rng.random() < 0.25:
                    L.rect(wx, wy, wx + 0.08, wy + 0.12, "#e8b04a")
        x += bw + rng.uniform(0.02, 0.12)
    L.rect(0, base, W, BODY_B, navy)
    L.text(FRONT, zy(86), "FIVE", "serif", 0.75, navy, "mm", tracking=0.3)
    L.text(FRONT, zy(76), "BOROUGHS", "serif", 1.0, navy, "mm", tracking=0.05)
    L.rect(FRONT - 3.2, zy(69.5), FRONT + 3.2, zy(69.0), "#b8862f")
    L.text(FRONT, zy(65), "L A G E R", "sans", 0.45, "#b8862f", "mm", tracking=0.2)
    L.text(FRONT, zy(25), "CRISP  •  CLEAN  •  CITY-BREWED", "sans", 0.22, cream, "mm", tracking=0.12)
    L.rect(BACK - 2.9, BODY_T + 0.2, BACK + 2.9, zy(42), "#ffffff", radius=0.15)
    y = L.para(BACK - 2.7, BODY_T + 0.5, 5.4, "Brewed and canned by Five Boroughs Brewing Co., Ridgewood, NY 11385. "
               "A golden lager with a soft malt body and a clean, snappy finish.", "serif", 0.2, navy, 1.2)
    y = L.para(BACK - 2.7, y + 0.2, 5.4, "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT DRINK "
               "ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC "
               "BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS.", "sans", 0.15, navy, 1.18)
    L.barcode(BACK - 1.5, y + 0.3, 3.0, 1.0, digits="085431200917", color="#111111")
    return L, finish(tab=(0.80, 0.58, 0.26), label_rough=0.32)


def fizz():
    """Seltzer: mostly bare metal with fine white + lime ink, the minimal look."""
    L = Label(W, TOT, PPC)
    L.rect(0, TOP, W, BOT, (1, 1, 1, ), )
    L.im.putalpha(0)
    lime = "#7fd13b"
    for k in range(9):
        L.rect(0, zy(30 + k * 6.5), W, zy(30 + k * 6.5) + 0.05, lime)
    L.text(FRONT, zy(78), "fizz", "round", 1.9, "#14361f", "mm", tracking=-0.02)
    L.text(FRONT, zy(58), "LIME  SPARKLING WATER", "sans", 0.32, lime, "mm", tracking=0.18)
    L.ellipse(FRONT, zy(45), 0.9, 0.9, lime)
    L.text(FRONT, zy(45), "0", "round", 0.85, "#ffffff", "mm")
    L.text(FRONT, zy(39), "CALORIES • SWEETENERS • SODIUM", "sans", 0.17, "#14361f", "mm", tracking=0.1)
    L.text(W / 2, (TOP + zy(105.5)) / 2, "fizz  ·  " * 14, "round", 0.3, "#14361f", "mm", tracking=0.2)
    L.text(FRONT, (BODY_B + BOT) / 2, "12 FL OZ (355 mL)", "sans", 0.26, "#14361f", "mm", tracking=0.15)
    L.rect(BACK - 2.9, BODY_T + 0.2, BACK + 2.9, BODY_B - 0.05, "#ffffff", radius=0.15)
    info_panel(L, BACK, 0, 0, 0, "Fizz Water Co., Brooklyn, NY 11222", "CARBONATED WATER, NATURAL LIME FLAVOR.", ink="#2f6b12")
    return L, finish(label_rough=0.18)


def front_stoop():
    L = Label(W, TOT, PPC)
    amber, brown, cream = "#c9772b", "#4a2a17", "#f6e7c8"
    L.gradient(0, TOP, W, BOT, "#e09a3e", "#a85a1d")
    bands(L, None, "FRONT STOOP  •  BREWED SWEET TEA  •  LEMON", "12 FL OZ (355 mL)", ink=cream, neck_color=brown)
    sx, gy = FRONT - 3.3, zy(16)                                      # brownstone: facade, arched door, stoop + railing
    L.rect(sx - 0.6, zy(74), sx + 6.0, gy, "#7a4426")
    for k in range(4):                                                # window pairs
        for wx in (sx - 0.2, sx + 4.3):
            L.rect(wx, zy(70 - k * 13), wx + 1.0, zy(62 - k * 13), "#2b1a10")
            L.rect(wx - 0.08, zy(62 - k * 13), wx + 1.08, zy(61 - k * 13), cream)
    L.rect(sx - 0.8, zy(75), sx + 6.2, zy(73), brown)                 # cornice
    L.rect(sx + 1.9, zy(52), sx + 3.6, zy(36), "#2b1a10")
    L.ellipse(sx + 2.75, zy(52), 0.85, 0.6, "#2b1a10")
    L.rect(sx + 2.05, zy(50), sx + 3.45, zy(37), "#f0b35a")
    L.ellipse(sx + 2.75, zy(50), 0.7, 0.5, "#f0b35a")
    for k in range(8):                                                # stoop treads coming toward us
        y0 = zy(36 - k * 2.5)
        L.rect(sx + 1.4 - k * 0.32, y0, sx + 4.1 + k * 0.32, zy(36 - (k + 1) * 2.5), brown if k % 2 else "#5c3520")
    for side in (-1, 1):                                              # iron railings
        x0 = sx + 2.75 + side * 1.45
        L.line([(x0, zy(38)), (x0 + side * 2.6, zy(16))], "#120c08", 0.07)
        L.line([(x0, zy(42)), (x0 + side * 2.6, zy(20))], "#120c08", 0.05)
        for t in np.linspace(0, 1, 7):
            L.line([(x0 + side * 2.6 * t, zy(38 - 22 * t)), (x0 + side * 2.6 * t, zy(42 - 22 * t))], "#120c08", 0.03)
    L.text(FRONT + 0.4, zy(97), "Front Stoop", "serif_i", 0.95, cream, "mm")
    L.text(FRONT + 0.4, zy(90), "SWEET TEA WITH LEMON", "sans", 0.25, cream, "mm", tracking=0.14)
    L.rect(BACK - 2.9, BODY_T + 0.2, BACK + 2.9, BODY_B - 0.05, cream, radius=0.15)
    info_panel(L, BACK, 110, 28, 22, "Front Stoop Beverages, Bronx, NY 10451", "BREWED TEA (FILTERED WATER, BLACK TEA), CANE SUGAR, "
               "LEMON JUICE CONCENTRATE, CITRIC ACID, NATURAL FLAVOR.", ink=brown, bg=cream)
    return L, finish(tab=(0.62, 0.38, 0.16), label_rough=0.28)


VARIANTS = {"Kolossal": kolossal, "Zing": zing, "Volt": volt, "FiveBoroughs": lager, "Fizz": fizz, "FrontStoop": front_stoop}


def to_uv(L, wear_seed=None):
    """Place a label into the 2048² UV space (side island) as RGBA float."""
    art = L.array()
    if wear_seed is not None:
        art = age(art, wear_seed, fade=0.04, scuff=0.05)
    full = np.zeros((2048, 2048, 4), np.float32)
    u0, v0, u1, v1 = G.UV_SIDE
    i0, i1 = int(round((1 - v1) * 2048)), int(round((1 - v0) * 2048))
    from PIL import Image
    a = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((2048, i1 - i0), Image.LANCZOS), np.float32) / 255
    full[i0:i1] = a
    return full
