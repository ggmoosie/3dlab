"""pxgrid.py - zoomed, labelled pixel grid of a photo crop (read landmarks off the reference).
usage: .venv/bin/python guns/ak74m/pxgrid.py <photo> x0 y0 x1 y1 zoom step out.png"""
import sys
from PIL import Image, ImageDraw
f, x0, y0, x1, y1, k, step, out = sys.argv[1], *map(int, sys.argv[2:8]), sys.argv[8]
im = Image.open(f).convert("RGB")
c = im.crop((x0, y0, x1, y1)).resize(((x1 - x0) * k, (y1 - y0) * k), Image.NEAREST if k >= 6 else Image.LANCZOS)
d = ImageDraw.Draw(c)
for x in range(x0 - x0 % step + step, x1, step):
    X = (x - x0) * k; d.line([(X, 0), (X, c.height)], fill=(255, 0, 0)); d.text((X + 2, 2), str(x), fill=(255, 0, 0))
for y in range(y0 - y0 % step + step, y1, step):
    Y = (y - y0) * k; d.line([(0, Y), (c.width, Y)], fill=(0, 0, 255)); d.text((2, Y + 1), str(y), fill=(0, 0, 255))
c.save(out)
