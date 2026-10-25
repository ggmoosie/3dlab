"""pxgrid.py - zoomed, labelled pixel grid of a photo crop (read coordinates off the reference; output stays in work/).
usage: python pxgrid.py photo x0 y0 x1 y1 step out.png [zoom]"""
import sys
from PIL import Image, ImageDraw, ImageFont
p, x0, y0, x1, y1, st, out = sys.argv[1], *map(int, sys.argv[2:7]), sys.argv[7]
z = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0
im = Image.open(p).convert("RGB").crop((x0, y0, x1, y1))
im = im.resize((int(im.width * z), int(im.height * z)), Image.LANCZOS)
d = ImageDraw.Draw(im)
f = ImageFont.truetype(__import__("os").path.join(__import__("os").path.dirname(__file__), "../../fonts/WorkSans-Bold.ttf"), 13)
for x in range((x0 // st + 1) * st, x1, st):
    X = (x - x0) * z; d.line([(X, 0), (X, im.height)], fill=(255, 0, 255) if x % (st * 5) == 0 else (0, 200, 255), width=1)
    d.text((X + 2, 2), str(x), fill=(220, 0, 220), font=f)
for y in range((y0 // st + 1) * st, y1, st):
    Y = (y - y0) * z; d.line([(0, Y), (im.width, Y)], fill=(255, 0, 255) if y % (st * 5) == 0 else (0, 200, 255), width=1)
    d.text((2, Y + 2), str(y), fill=(220, 0, 220), font=f)
im.save(out)
