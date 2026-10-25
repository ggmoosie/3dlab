"""Packaging/label design kit (PIL). Lay out in real centimetres; rasterised at px_per_cm with 2x supersampling.
All brands are invented. No logos of real companies, ever."""
import os, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
F = {
    "display": "BigShoulders-Bold.ttf", "display_r": "BigShoulders-Regular.ttf", "grotesk": "BricolageGrotesque-Bold.ttf",
    "grotesk_r": "BricolageGrotesque-Regular.ttf", "round": "Outfit-Bold.ttf", "round_r": "Outfit-Regular.ttf",
    "sans": "WorkSans-Bold.ttf", "sans_r": "WorkSans-Regular.ttf", "sans_i": "WorkSans-BoldItalic.ttf",
    "script": "EricaOne-Regular.ttf", "serif": "YoungSerif-Regular.ttf", "serif_b": "Lora-Bold.ttf",
    "serif_i": "Lora-BoldItalic.ttf", "hand": "NothingYouCouldDo-Regular.ttf", "mono": "IBMPlexMono-Regular.ttf",
    "mono_b": "IBMPlexMono-Bold.ttf", "stencil": "NationalPark-Bold.ttf", "tech": "Tektur-Medium.ttf", "heavy": "Boldonse-Regular.ttf",
}


def rgba(c, a=1.0):
    if isinstance(c, str):
        c = c.lstrip("#")
        c = tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))
    elif max(c) <= 1.0:
        if len(c) == 4:
            a = c[3]
        c = tuple(int(v * 255) for v in c[:3])
    return (*c[:3], int(a * 255))


class Label:
    """A print sheet w_cm × h_cm. Coordinates in cm, origin top-left."""

    def __init__(self, w_cm, h_cm, px_per_cm=40, bg=None, ss=2):
        self.w, self.h, self.ppc, self.ss = w_cm, h_cm, px_per_cm, ss
        self.W, self.H = int(round(w_cm * px_per_cm)), int(round(h_cm * px_per_cm))
        self.im = Image.new("RGBA", (self.W * ss, self.H * ss), rgba(bg) if bg else (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def p(self, x, y):
        k = self.ppc * self.ss
        return (x * k, y * k)

    def s(self, v):
        return v * self.ppc * self.ss

    def font(self, key, size_cm):
        return ImageFont.truetype(os.path.join(FONTS, F.get(key, key)), max(4, int(self.s(size_cm))))

    # ------------------------------------------------------------ shapes
    def rect(self, x0, y0, x1, y1, fill, radius=0, outline=None, width=0):
        box = [*self.p(x0, y0), *self.p(x1, y1)]
        if radius:
            self.d.rounded_rectangle(box, self.s(radius), fill=rgba(fill) if fill else None,
                                     outline=rgba(outline) if outline else None, width=int(self.s(width)))
        else:
            self.d.rectangle(box, fill=rgba(fill) if fill else None, outline=rgba(outline) if outline else None, width=int(self.s(width)))

    def ellipse(self, cx, cy, rx, ry, fill, outline=None, width=0):
        self.d.ellipse([*self.p(cx - rx, cy - ry), *self.p(cx + rx, cy + ry)], fill=rgba(fill) if fill else None,
                       outline=rgba(outline) if outline else None, width=int(self.s(width)))

    def poly(self, pts, fill):
        self.d.polygon([self.p(*q) for q in pts], fill=rgba(fill))

    def line(self, pts, color, width):
        self.d.line([self.p(*q) for q in pts], fill=rgba(color), width=max(1, int(self.s(width))), joint="curve")

    def stripes(self, x0, y0, x1, y1, color, pitch, width, angle=0):
        """Parallel stripes clipped to a rect (angle in degrees)."""
        layer = Image.new("L", self.im.size, 0)
        d = ImageDraw.Draw(layer)
        a = math.radians(angle)
        ca, sa = math.cos(a), math.sin(a)
        L = (x1 - x0) + (y1 - y0)
        t = -L
        while t < L * 2:
            cx, cy = x0 + t, y0
            d.line([self.p(cx - sa * L, cy - ca * -L), self.p(cx + sa * L, cy + ca * L)] if angle else
                   [self.p(x0 + t, y0), self.p(x0 + t, y1)], fill=255, width=max(1, int(self.s(width))))
            t += pitch
        clip = Image.new("L", self.im.size, 0)
        ImageDraw.Draw(clip).rectangle([*self.p(x0, y0), *self.p(x1, y1)], fill=255)
        layer = Image.fromarray(np.minimum(np.asarray(layer), np.asarray(clip)))
        self.im.paste(Image.new("RGBA", self.im.size, rgba(color)), (0, 0), layer)

    def gradient(self, x0, y0, x1, y1, c0, c1, vertical=True):
        w, h = int(self.s(x1 - x0)), int(self.s(y1 - y0))
        t = np.linspace(0, 1, h if vertical else w)
        a, b = np.array(rgba(c0), float), np.array(rgba(c1), float)
        g = a[None] * (1 - t[:, None]) + b[None] * t[:, None]
        arr = np.repeat(g[:, None, :], w, 1) if vertical else np.repeat(g[None, :, :], h, 0)
        self.im.alpha_composite(Image.fromarray(arr.astype(np.uint8), "RGBA"), (int(self.s(x0)), int(self.s(y0))))

    def halftone(self, x0, y0, x1, y1, color, pitch, rmax, fade="x"):
        for i, x in enumerate(np.arange(x0, x1, pitch)):
            for j, y in enumerate(np.arange(y0, y1, pitch)):
                t = (x - x0) / (x1 - x0) if fade == "x" else (y - y0) / (y1 - y0)
                r = rmax * (1 - t)
                if r > 0.002:
                    self.ellipse(x + (pitch / 2 if j % 2 else 0), y, r, r, color)

    def burst(self, cx, cy, r0, r1, n, color, rot=0):
        pts = []
        for k in range(n * 2):
            a = math.radians(rot) + k * math.pi / n
            r = r1 if k % 2 == 0 else r0
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        self.poly(pts, color)

    # ------------------------------------------------------------ type
    def text(self, x, y, s, key, size, color, anchor="la", tracking=0.0, max_w=None, stroke=0, stroke_color=None, rotate=0):
        """Draw text with optional letter tracking (em fraction) and auto-shrink to max_w (cm)."""
        while True:
            f = self.font(key, size)
            w = self._width(s, f, tracking)
            if not max_w or w <= self.s(max_w) or size < 0.05:
                break
            size *= 0.95
        if rotate:
            pad = int(self.s(size))
            tmp = Image.new("RGBA", (int(w) + pad * 2, int(self.s(size) * 1.6) + pad * 2), (0, 0, 0, 0))
            Label._draw_tracked(ImageDraw.Draw(tmp), (pad, pad), s, f, tracking, color, "la", stroke, stroke_color, self)
            tmp = tmp.rotate(rotate, expand=True, resample=Image.BICUBIC)
            px, py = self.p(x, y)
            self.im.alpha_composite(tmp, (int(px - tmp.width / 2), int(py - tmp.height / 2)))
            return size
        Label._draw_tracked(self.d, self.p(x, y), s, f, tracking, color, anchor, stroke, stroke_color, self)
        return size

    def _width(self, s, f, tracking):
        return sum(f.getlength(c) for c in s) + tracking * f.size * max(len(s) - 1, 0)

    @staticmethod
    def _draw_tracked(d, xy, s, f, tracking, color, anchor, stroke, stroke_color, lab):
        x, y = xy
        if tracking == 0:
            d.text((x, y), s, font=f, fill=rgba(color), anchor=anchor, stroke_width=int(lab.s(stroke)),
                   stroke_fill=rgba(stroke_color) if stroke_color else None)
            return
        w = lab._width(s, f, tracking)
        if anchor[0] == "m":
            x -= w / 2
        elif anchor[0] == "r":
            x -= w
        for c in s:
            d.text((x, y), c, font=f, fill=rgba(color), anchor="l" + anchor[1], stroke_width=int(lab.s(stroke)),
                   stroke_fill=rgba(stroke_color) if stroke_color else None)
            x += f.getlength(c) + tracking * f.size

    def para(self, x, y, w, s, key, size, color, leading=1.25):
        """Wrapped paragraph in a column w cm wide. Returns the y below it."""
        f = self.font(key, size)
        line, lines = "", []
        for word in s.split():
            t = (line + " " + word).strip()
            if f.getlength(t) > self.s(w) and line:
                lines.append(line)
                line = word
            else:
                line = t
        lines.append(line)
        for ln in lines:
            self.d.text(self.p(x, y), ln, font=f, fill=rgba(color))
            y += size * leading
        return y

    # ------------------------------------------------------------ packaging furniture
    def barcode(self, x, y, w, h, digits="0123456789012", color="#111111", bg="#ffffff"):
        """EAN-13 with a correct check digit and guard bars."""
        L = ["0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011"]
        G = ["".join("1" if c == "0" else "0" for c in s)[::-1] for s in L]
        R = ["".join("1" if c == "0" else "0" for c in s) for s in L]
        par = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]
        d = [int(c) for c in digits[:12]]
        chk = (10 - (sum(d[i] * (3 if i % 2 else 1) for i in range(12)) % 10)) % 10
        d.append(chk)
        bits = "101" + "".join((L if p == "L" else G)[d[i + 1]] for i, p in enumerate(par[d[0]])) + "01010" + \
               "".join(R[k] for k in d[7:]) + "101"
        self.rect(x - w * 0.08, y - h * 0.05, x + w * 1.04, y + h * 1.22, bg)
        mw = w / len(bits)
        for i, b in enumerate(bits):
            if b == "1":
                guard = i < 3 or 45 <= i < 50 or i >= 92
                self.rect(x + i * mw, y, x + (i + 1) * mw, y + h * (1.08 if guard else 1.0), color)
        f = h * 0.16
        self.text(x - w * 0.06, y + h * 1.03, str(d[0]), "mono", f, color)
        self.text(x + w * 0.27, y + h * 1.03, "".join(map(str, d[1:7])), "mono", f, color, "ma", tracking=0.25)
        self.text(x + w * 0.73, y + h * 1.03, "".join(map(str, d[7:])), "mono", f, color, "ma", tracking=0.25)

    def nutrition(self, x, y, w, rows, title="Nutrition Facts", serving="Serving size 1 can (355 mL)", color="#111111", bg="#ffffff"):
        """A generic nutrition panel. rows: [(name, amount, pct or ''), ...]. Returns bottom y."""
        lh = w * 0.075
        h = lh * (len(rows) + 5.2)
        self.rect(x, y, x + w, y + h, bg, outline=color, width=w * 0.012)
        p = w * 0.04
        self.text(x + p, y + p * 0.4, title, "sans", lh * 1.25, color, max_w=w - 2 * p)
        cy = y + lh * 1.75
        self.rect(x + p, cy, x + w - p, cy + w * 0.006, color)
        self.text(x + p, cy + lh * 0.15, serving, "sans_r", lh * 0.6, color, max_w=w - 2 * p)
        cy += lh * 1.0
        self.rect(x + p, cy, x + w - p, cy + w * 0.03, color)
        cy += lh * 0.45
        self.text(x + w - p, cy, "% Daily Value*", "sans", lh * 0.55, color, "ra")
        cy += lh * 0.75
        for name, amt, pct in rows:
            self.rect(x + p, cy - lh * 0.12, x + w - p, cy - lh * 0.12 + w * 0.004, color)
            bold = not name.startswith(" ")
            self.text(x + p + (0 if bold else lh * 0.6), cy, f"{name.strip()} {amt}", "sans" if bold else "sans_r", lh * 0.62, color)
            if pct:
                self.text(x + w - p, cy, pct, "sans", lh * 0.62, color, "ra")
            cy += lh
        self.rect(x + p, cy - lh * 0.12, x + w - p, cy - lh * 0.12 + w * 0.012, color)
        self.para(x + p, cy, w - 2 * p, "*The % Daily Value tells you how much a nutrient in a serving contributes to a daily diet.",
                  "sans_r", lh * 0.42, color, 1.15)
        return y + h

    # ------------------------------------------------------------ output
    def image(self):
        return self.im.resize((self.W, self.H), Image.LANCZOS)

    def array(self):
        return np.asarray(self.image(), np.float32) / 255.0

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.image().save(path)
        return path


def age(arr, seed=0, fade=0.12, scuff=0.25, rough_px=None):
    """Age a print (RGBA float): ink fade toward paper, scuffed alpha along random rub marks, fine speckle loss."""
    from . import tex
    h, w = arr.shape[:2]
    out = arr.copy()
    n = tex.fbm(h, w, 6, seed)
    out[..., :3] = out[..., :3] * (1 - fade * n[..., None]) + fade * n[..., None] * 0.92
    rub = tex.scratches(h, w, n=int(300 * scuff), length=(0.005, 0.03), width=1, seed=seed + 3)
    loss = np.clip(rub * scuff * 6 * (n > 0.55) + tex.speckle(h, w, 0.00012, seed + 4) * scuff * 2, 0, 1)
    out[..., 3] *= 1 - loss
    return out
