"""Glass bottle prints (invented brands): front label 7.5 × 8.5 cm, neck foil band 8.4 × 3.8 cm (wrap), crown top Ø3.36 cm.
Each variant → (front, neck, cap Labels, params) incl. glass + liquid colours. Neck ink sits over the foil (layer B)."""
import math, os, sys, zlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
from lab.labels import Label, age
import glass_geo as G

LW, LH = 7.5, 8.5
NW, NH = 2 * math.pi * 1.34, 3.8
CD = 3.36
PPC = 110
AMBER, GREEN, CLEAR = (0.55, 0.30, 0.07), (0.30, 0.55, 0.28), (0.95, 0.97, 0.96)


def params(glass, liquid, murk, cap, cap_metal, foil, foil_metal=1.0, label_rough=0.6):
    return dict(ColorA=cap, RoughA=0.32, MetalA=cap_metal, ColorB=foil, RoughB=0.28, MetalB=foil_metal,
                LabelRough=label_rough, GlassTint=glass, LiquidTint=liquid, LiquidMurk=murk)


def legal(L, x, y, w, text, ink, abv=None):
    if abv:
        L.text(x + w / 2, y, f"12 FL OZ  ·  {abv}% ALC/VOL", "sans", 0.2, ink, "mm", tracking=0.08)
    L.para(x, y + 0.3, w, text, "sans_r", 0.12, ink, 1.15)
    L.barcode(x + w - 1.7, LH - 1.35, 1.6, 0.8, "0" + str(zlib.crc32(text.encode()) % 10 ** 11).zfill(11))


WARN = "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. (2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS."


def cap_print(text, ink, ring=None, size=0.55):
    C = Label(CD, CD, PPC)
    if ring:
        C.ellipse(CD / 2, CD / 2, 1.3, 1.3, None, ring, 0.08)
    C.text(CD / 2, CD / 2, text, "heavy", size, ink, "mm")
    return C


def gowanus():
    kraft, green, ink = "#c8a678", "#2f5a2a", "#2a1d12"
    F = Label(LW, LH, PPC, bg=kraft)
    F.rect(0.25, 0.25, LW - 0.25, LH - 0.25, None, outline=ink, width=0.04)
    for k in range(7):                                             # hop cone from overlapping scales
        for j in range(k + 1):
            x, y = LW / 2 + (j - k / 2) * 0.42, 1.0 + k * 0.32
            F.ellipse(x, y, 0.28, 0.24, green if (j + k) % 2 else "#4d7a3a", outline=ink, width=0.015)
    F.text(LW / 2, 4.1, "GOWANUS", "display", 1.05, ink, "mm", tracking=0.08)
    F.text(LW / 2, 5.0, "PALE ALE", "serif", 0.55, green, "mm", tracking=0.18)
    F.text(LW / 2, 5.65, "CANAL-SIDE BREWING · BROOKLYN", "sans", 0.18, ink, "mm", tracking=0.1)
    legal(F, 0.5, 6.15, LW - 1.0, "Brewed & bottled by Gowanus Canal Brewing Co., Brooklyn, NY 11215. " + WARN, ink, 5.6)
    N = Label(NW, NH, PPC)
    for k in range(3):
        N.text(NW * (k + 0.5) / 3, NH / 2, "GOWANUS", "display", 0.75, ink, "mm", tracking=0.1)
    return F, N, cap_print("G", "#f0e6d0", "#f0e6d0"), params(AMBER, (0.85, 0.55, 0.15), 0.12, (0.12, 0.25, 0.10), 0.6, (0.75, 0.60, 0.30))


def ironworks():
    black, copper, cream = "#141414", "#b87333", "#e9dcc4"
    F = Label(LW, LH, PPC, bg=black)
    cx, cy = LW / 2, 2.0
    for k in range(12):                                             # gear
        a = 2 * math.pi * k / 12
        F.poly([(cx + 1.25 * math.cos(a - 0.12), cy + 1.25 * math.sin(a - 0.12)), (cx + 1.55 * math.cos(a - 0.08), cy + 1.55 * math.sin(a - 0.08)),
                (cx + 1.55 * math.cos(a + 0.08), cy + 1.55 * math.sin(a + 0.08)), (cx + 1.25 * math.cos(a + 0.12), cy + 1.25 * math.sin(a + 0.12))], copper)
    F.ellipse(cx, cy, 1.3, 1.3, copper); F.ellipse(cx, cy, 0.95, 0.95, black)
    F.text(cx, cy, "IW", "stencil", 0.75, copper, "mm")
    F.text(LW / 2, 4.3, "IRONWORKS", "stencil", 0.85, cream, "mm", tracking=0.06)
    F.text(LW / 2, 5.15, "OATMEAL STOUT", "sans", 0.36, copper, "mm", tracking=0.25)
    legal(F, 0.5, 5.75, LW - 1.0, "Ironworks Brewing, Red Hook, Brooklyn, NY 11231. " + WARN, cream, 6.8)
    N = Label(NW, NH, PPC, bg=None)
    N.text(NW / 2, NH / 2, "IRONWORKS  ·  STOUT  ·  IRONWORKS", "stencil", 0.42, black, "mm", tracking=0.1)
    return F, N, cap_print("IW", copper, copper, 0.45), params(AMBER, (0.07, 0.035, 0.02), 0.0, (0.03, 0.03, 0.03), 0.4, (0.72, 0.40, 0.18))


def coney():
    F = Label(LW, LH, PPC, bg="#f7f2e6")
    for k in range(10):                                            # carousel canopy stripes
        F.poly([(LW / 2, 0.6), (k * LW / 10, 2.6), ((k + 1) * LW / 10, 2.6)], "#c8202f" if k % 2 else "#f7f2e6")
    F.rect(0, 2.6, LW, 2.85, "#1d3f8f")
    F.text(LW / 2, 3.75, "Coney", "script", 1.1, "#c8202f", "mm", stroke=0.03, stroke_color="#1d3f8f")
    F.text(LW / 2, 4.85, "BOARDWALK LAGER", "sans", 0.36, "#1d3f8f", "mm", tracking=0.16)
    F.text(LW / 2, 5.45, "EST. ON THE SHORE", "sans_r", 0.18, "#1d3f8f", "mm", tracking=0.2)
    legal(F, 0.5, 5.9, LW - 1.0, "Coney Shore Brewers, Brooklyn, NY 11224. " + WARN, "#1d3f8f", 4.5)
    N = Label(NW, NH, PPC)
    N.text(NW / 2, NH / 2, "CONEY  ★  LAGER  ★  CONEY", "sans", 0.4, "#ffffff", "mm", tracking=0.12)
    return F, N, cap_print("★", "#ffffff", "#ffffff", 0.9), params(GREEN, (0.9, 0.7, 0.3), 0.08, (0.6, 0.06, 0.08), 0.5, (0.12, 0.22, 0.55))


def lemonade():
    F = Label(LW, LH, PPC, bg="#fff6c9")
    for i in range(16):                                            # gingham
        F.rect(i * LW / 16, 0, i * LW / 16 + LW / 32, LH, (1.0, 0.82, 0.15, 0.45))
    for j in range(18):
        F.rect(0, j * LH / 18, LW, j * LH / 18 + LH / 36, (1.0, 0.82, 0.15, 0.45))
    F.rect(0.6, 2.2, LW - 0.6, 5.6, "#fffdf2", radius=0.4, outline="#e0a400", width=0.05)
    F.text(LW / 2, 3.0, "Grandma Jo's", "serif_i", 0.75, "#7a4b00", "mm")
    F.text(LW / 2, 4.0, "OLD-FASHIONED", "sans", 0.26, "#c27c00", "mm", tracking=0.2)
    F.text(LW / 2, 4.7, "LEMONADE", "heavy", 0.48, "#7a4b00", "mm", tracking=0.08)
    F.para(0.7, 6.0, LW - 1.4, "Fresh-squeezed lemons, cane sugar & spring water. Shake gently. Refrigerate after opening. Jo's Kitchen, Astoria, NY 11102.", "sans_r", 0.16, "#7a4b00", 1.15)
    F.barcode(LW - 2.3, LH - 1.3, 1.6, 0.8, "076543219876")
    N = Label(NW, NH, PPC)
    N.text(NW / 2, NH / 2, "since 1958 · since 1958", "serif_i", 0.5, "#7a4b00", "mm")
    return F, N, cap_print("Jo", "#7a4b00", "#7a4b00", 0.6), params(CLEAR, (0.97, 0.92, 0.6), 0.45, (0.95, 0.78, 0.1), 0.3, (0.95, 0.85, 0.40), label_rough=0.55)


def kolossal_glass():
    F = Label(LW, LH, PPC, bg="#c8102e")
    for k in range(60):
        x = LW * k / 59
        F.ellipse(x, 5.0 + 0.45 * math.sin(x * 1.1), 0.1, 0.26, "#ffffff")
    F.text(LW / 2, 2.6, "Kolossal", "script", 1.0, "#ffffff", "mm", rotate=5)
    F.text(LW / 2, 3.75, "C O L A", "heavy", 0.4, "#ffffff", "mm")
    F.text(LW / 2, 6.0, "CLASSIC GLASS · REAL CANE SUGAR", "sans", 0.18, "#ffd9df", "mm", tracking=0.1)
    F.para(0.5, 6.4, LW - 1.0, "12 FL OZ (355 mL). Kolossal Beverage Co., Long Island City, NY 11101. Return for deposit NY 5¢.", "sans_r", 0.14, "#ffffff", 1.15)
    F.barcode(LW - 2.2, LH - 1.25, 1.6, 0.75, "049000050103")
    N = Label(NW, NH, PPC)
    N.text(NW / 2, NH / 2, "ICE COLD · KOLOSSAL · ICE COLD", "sans", 0.36, "#ffffff", "mm", tracking=0.12)
    return F, N, cap_print("K", "#ffffff", "#ffffff", 0.9), params(CLEAR, (0.22, 0.07, 0.03), 0.05, (0.75, 0.05, 0.06), 0.5, (0.75, 0.05, 0.06), 0.6, 0.45)


def cider():
    F = Label(LW, LH, PPC, bg="#f3ead3")
    F.ellipse(LW / 2, 2.0, 1.15, 1.05, "#b8262b")                    # apple
    F.ellipse(LW / 2 - 0.35, 1.7, 0.3, 0.2, "#e05a50")
    F.poly([(LW / 2, 1.0), (LW / 2 + 0.1, 0.55), (LW / 2 + 0.2, 0.6), (LW / 2 + 0.08, 1.02)], "#4a2f1a")
    F.poly([(LW / 2 + 0.1, 0.75), (LW / 2 + 0.75, 0.45), (LW / 2 + 0.55, 0.9)], "#4b7a2c")
    F.text(LW / 2, 3.7, "Hudson Valley", "serif_i", 0.62, "#3d2a1a", "mm")
    F.text(LW / 2, 4.6, "HARD CIDER", "sans", 0.42, "#b8262b", "mm", tracking=0.2)
    F.text(LW / 2, 5.25, "ORCHARD DRY · UNFILTERED", "sans_r", 0.18, "#3d2a1a", "mm", tracking=0.12)
    legal(F, 0.5, 5.7, LW - 1.0, "Pressed & bottled at Hudson Valley Cider Mill, Kingston, NY 12401. " + WARN, "#3d2a1a", 6.2)
    N = Label(NW, NH, PPC)
    N.text(NW / 2, NH / 2, "HUDSON VALLEY · CIDER", "serif_i", 0.42, "#3d2a1a", "mm")
    return F, N, cap_print("HV", "#3d2a1a", "#3d2a1a", 0.5), params(GREEN, (0.92, 0.66, 0.25), 0.2, (0.80, 0.62, 0.25), 1.0, (0.80, 0.62, 0.25))


VARIANTS = {"GowanusPale": gowanus, "IronworksStout": ironworks, "ConeyLager": coney, "GrandmaJoLemonade": lemonade,
            "KolossalGlass": kolossal_glass, "HudsonCider": cider}


def to_uv(F, N, C, seed):
    full = np.zeros((2048, 2048, 4), np.float32)
    for lab, rect, sc in ((F, G.UV_LABEL, 0.06), (N, G.UV_NECK, 0.12), (C, G.UV_CAP_TOP, 0.2)):
        art = age(lab.array(), seed, fade=0.05, scuff=sc)
        u0, v0, u1, v1 = rect
        j0, j1, i0, i1 = int(round(u0 * 2048)), int(round(u1 * 2048)), int(round((1 - v1) * 2048)), int(round((1 - v0) * 2048))
        if rect is G.UV_CAP_TOP:                                    # disc island: the print sits inside the rim
            pad = int((j1 - j0) * (1 - CD / 3.36) / 2)
            j0, j1, i0, i1 = j0 + pad, j1 - pad, i0 + pad, i1 - pad
        full[i0:i1, j0:j1] = np.asarray(Image.fromarray((art * 255).astype(np.uint8)).resize((j1 - j0, i1 - i0), Image.LANCZOS), np.float32) / 255
    return full
