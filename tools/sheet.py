"""Contact sheet: python tools/sheet.py out.jpg title img1 img2 ... (2 per row, labelled with file names)."""
import sys, os
from PIL import Image, ImageDraw, ImageFont
out, title, ims = sys.argv[1], sys.argv[2], sys.argv[3:]
W = 1000
tiles = []
for p in ims:
    im = Image.open(p).convert("RGB")
    im = im.resize((W, int(im.height * W / im.width)))
    tiles.append((os.path.basename(p), im))
rows = [tiles[i:i + 2] for i in range(0, len(tiles), 2)]
H = 70 + sum(max(t[1].height for t in r) + 36 for r in rows)
sheet = Image.new("RGB", (W * 2 + 30, H), (22, 23, 26))
d = ImageDraw.Draw(sheet)
f = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "../fonts/WorkSans-Bold.ttf"), 34)
fs = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "../fonts/WorkSans-Regular.ttf"), 20)
d.text((16, 16), title, font=f, fill=(235, 235, 235))
y = 70
for r in rows:
    for i, (n, im) in enumerate(r):
        sheet.paste(im, (10 + i * (W + 10), y + 30))
        d.text((14 + i * (W + 10), y + 4), n, font=fs, fill=(160, 165, 175))
    y += max(t[1].height for t in r) + 36
sheet.save(out, quality=88)
print(out)
