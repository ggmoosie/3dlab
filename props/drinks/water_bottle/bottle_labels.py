"""Label band prints for the 500 mL bottle (invented brands). Band = 20.11 × 3.5 cm wrap; x = 0.75 W faces -Y.
Each variant returns (Label, params) where params also carry the liquid tint and cap colour."""
import math, os, sys, zlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab.labels import Label, age
import bottle_geo as G

W, H = 2 * math.pi * 3.20, 3.5
PPC = 2048 / W
FRONT, BACK = 0.75 * W, 0.25 * W
WATER = (0.92, 0.96, 1.0)


def finish(cap, liquid=WATER, cap_rough=0.45, label_rough=0.35, murk=0.0):
    return dict(ColorA=cap, RoughA=cap_rough, MetalA=0.0, ColorB=(0.5, 0.5, 0.5), RoughB=0.4, MetalB=0.0,
                LabelRough=label_rough, LiquidTint=liquid, LiquidMurk=murk)


def back_panel(L, cx, company, extra="", ink="#1d2b4a"):
    w = 5.2
    y = L.para(cx - w / 2, 0.28, w, "Nutrition: Serving 500 mL · Calories 0 · Sodium 5mg (0% DV) · Not a significant source of other nutrients.", "sans", 0.13, ink, 1.15)
    y = L.para(cx - w / 2, y + 0.05, w, extra or "INGREDIENTS: WATER.", "sans_r", 0.13, ink, 1.15)
    L.para(cx - w / 2, y + 0.05, w, company + " · BPA FREE · PLEASE RECYCLE · NY 5¢", "sans_r", 0.12, ink, 1.15)
    L.barcode(cx + w / 2 + 0.4, 0.55, 2.4, 1.6, "0" + str(zlib.crc32(company.encode()) % 10 ** 11).zfill(11))


def hudson():
    L = Label(W, H, PPC, bg="#ffffff")
    navy, sky = "#14335a", "#7fb6e0"
    L.gradient(0, 0, W, H, "#e9f4fc", "#ffffff")
    for k in range(5):                                            # river bands
        y = 2.5 + k * 0.17
        L.poly([(x, y + 0.12 * math.sin(x * 1.3 + k)) for x in np.linspace(0, W, 60)] + [(W, H), (0, H)], "#b8d9f0" if k % 2 else "#d9ecf8")
    for cx, h in ((FRONT - 2.3, 1.5), (FRONT - 1.0, 2.1), (FRONT + 0.6, 1.7), (BACK + 5.6, 1.4)):
        L.poly([(cx - 1.6, 2.6), (cx, 2.6 - h), (cx + 1.6, 2.6)], sky)          # mountains
        L.poly([(cx - 0.45, 2.6 - h + 0.42), (cx, 2.6 - h), (cx + 0.45, 2.6 - h + 0.42), (cx + 0.15, 2.6 - h + 0.32), (cx - 0.1, 2.6 - h + 0.45)], "#ffffff")
    L.text(FRONT, 0.55, "HUDSON SPRING", "display", 0.78, navy, "mm", tracking=0.06)
    L.text(FRONT, 1.15, "NATURAL SPRING WATER", "sans", 0.2, navy, "mm", tracking=0.22)
    L.text(FRONT, 3.12, "500 mL (16.9 FL OZ)", "sans", 0.17, navy, "mm", tracking=0.12)
    back_panel(L, BACK, "Bottled at the source by Hudson Spring Water Co., Catskill, NY 12414", ink=navy)
    return L, finish((0.03, 0.17, 0.55))


def pure_peak():
    L = Label(W, H, PPC)
    L.gradient(0, 0, W, H, "#5b3fa8", "#2b6fd6", vertical=False)
    for k in range(9):
        x = FRONT - 3 + k * 0.75
        L.poly([(x, 3.5), (x + 0.6, 2.75 - (k % 3) * 0.12), (x + 1.2, 3.5)], (1, 1, 1, 0.10 + 0.05 * (k % 2)))
    L.poly([(FRONT - 0.7, 1.55), (FRONT, 0.35), (FRONT + 0.7, 1.55)], "#ffffff")
    L.poly([(FRONT - 0.32, 1.55), (FRONT + 0.12, 0.8), (FRONT + 0.56, 1.55)], "#5b3fa8")
    L.text(FRONT, 1.95, "pure peak", "round", 0.62, "#ffffff", "mm")
    L.text(FRONT, 2.6, "PURIFIED DRINKING WATER  ·  pH 7.5", "sans", 0.16, "#dcd3ff", "mm", tracking=0.2)
    L.text(FRONT, 3.1, "500 mL", "sans", 0.16, "#ffffff", "mm", tracking=0.2)
    back_panel(L, BACK, "Pure Peak Waters, Paterson, NJ 07501", "INGREDIENTS: PURIFIED WATER, MAGNESIUM SULFATE, POTASSIUM BICARBONATE.", ink="#ffffff")
    return L, finish((0.85, 0.85, 0.87))


def value_fresh():
    L = Label(W, H, PPC, bg="#ffd320")
    L.rect(0, 0.0, W, 0.35, "#111111"); L.rect(0, 3.15, W, 3.5, "#111111")
    L.text(FRONT, 1.1, "DRINKING", "heavy", 0.55, "#111111", "mm", tracking=0.05)
    L.text(FRONT, 1.85, "WATER", "heavy", 0.55, "#111111", "mm", tracking=0.25)
    L.text(FRONT, 2.65, "VALUE FRESH  ·  SAME QUALITY, LOWER PRICE", "sans", 0.15, "#111111", "mm", tracking=0.1)
    L.text(W / 2, 0.18, "VALUE FRESH  ·  " * 10, "sans", 0.15, "#ffd320", "mm", tracking=0.2)
    back_panel(L, BACK, "Distributed by Value Fresh Markets, Secaucus, NJ 07094", ink="#111111")
    return L, finish((0.85, 0.85, 0.86))


def hydra(flavor="GLACIER", c0="#0a5bd6", c1="#29c4ff", liquid=(0.25, 0.65, 1.0), cap=(0.02, 0.2, 0.75)):
    def make():
        L = Label(W, H, PPC)
        L.gradient(0, 0, W, H, c0, c1, vertical=False)
        for k in range(14):                                       # speed slashes
            x = FRONT - 5 + k * 0.7
            L.poly([(x, 3.5), (x + 0.25, 3.5), (x + 1.4, 0), (x + 1.15, 0)], (1, 1, 1, 0.10))
        L.text(FRONT, 1.0, "HYDRA", "heavy", 0.8, "#ffffff", "mm", tracking=0.06, stroke=0.03, stroke_color="#0b1b40")
        L.text(FRONT, 1.85, f"{flavor}  ·  ELECTROLYTE SPORTS DRINK", "sans_i", 0.2, "#ffffff", "mm", tracking=0.06)
        L.rect(FRONT - 1.4, 2.35, FRONT + 1.4, 2.85, "#0b1b40", radius=0.12)
        L.text(FRONT, 2.6, "REHYDRATE · REPLENISH", "sans", 0.15, "#ffffff", "mm", tracking=0.12)
        L.text(FRONT, 3.18, "500 mL (16.9 FL OZ)", "sans", 0.15, "#ffffff", "mm", tracking=0.12)
        back_panel(L, BACK, "Hydra Performance Drinks, Hoboken, NJ 07030", "INGREDIENTS: WATER, SUGAR, DEXTROSE, CITRIC ACID, SALT, "
                   "SODIUM CITRATE, MONOPOTASSIUM PHOSPHATE, NATURAL FLAVOR, COLOR.", ink="#ffffff")
        return L, finish(cap, liquid, label_rough=0.25)
    return make


def used():
    """A survivor refill: the Hudson label sun-faded, torn, marker note; murky rainwater; a mismatched green cap."""
    L, _ = hudson()
    d = L.d
    rng = np.random.default_rng(4)
    tear = [(FRONT + 1.3, 0)] + [(FRONT + 1.3 + rng.uniform(-0.25, 0.25) + k * 0.32, k * 0.35) for k in range(1, 11)] + [(W, 3.5), (W, 0)]
    L.poly(tear, (0, 0, 0, 0))                                    # torn-away strip shows the bottle
    L.text(FRONT - 0.4, 2.15, "BOILED 4-12", "hand", 0.55, "#121212", "mm", rotate=-4)
    L.text(FRONT - 1.6, 2.95, "— D.", "hand", 0.4, "#121212", "mm", rotate=-4)
    L._aged = True
    return L, finish((0.08, 0.42, 0.12), liquid=(0.78, 0.74, 0.55), murk=0.35, label_rough=0.6)


VARIANTS = {"HudsonSpring": hudson, "PurePeak": pure_peak, "ValueFresh": value_fresh,
            "HydraGlacier": hydra(), "HydraFruitPunch": hydra("FRUIT PUNCH", "#c4122f", "#ff6a3d", (0.95, 0.18, 0.15), (0.75, 0.05, 0.05)),
            "Refill": used}


def to_uv(L, seed):
    art = L.array()
    heavy = getattr(L, "_aged", False)
    art = age(art, seed, fade=0.35 if heavy else 0.05, scuff=0.35 if heavy else 0.06)
    full = np.zeros((2048, 2048, 4), np.float32)
    u0, v0, u1, v1 = G.UV_LABEL
    i0, i1 = int(round((1 - v1) * 2048)), int(round((1 - v0) * 2048))
    from PIL import Image
    full[i0:i1] = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((2048, i1 - i0), Image.LANCZOS), np.float32) / 255
    return full
