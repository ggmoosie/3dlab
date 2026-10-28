---
name: softgoods-factory
description: Bang out game-ready soft gear (plate carriers, chest rigs, pouches, belts, packs, slings) in minutes, Tarkov style - a low-poly shell from flat panels plus procedural PBR detail (PALS webbing, bar-tacks, stitch rows, binding, loop fields, seams, wrinkles, grime) generated in texture space from the SAME panel layout. One spec file per item, one command per build. Use it for any fabric/nylon item; use photo-sdf-gun for hard-surface guns.
---

# Softgoods factory: layout -> low-poly + 2K PBR in one command

Reference item: `props/gear/plate_carrier/` (7 panels, 2.7k tris, 2K BaseColor/Normal/ORM, 3 texel/mm, build ~2 min).
Library: `lab/softgoods/core.py` (textures, pure numpy) and `lab/softgoods/panel.py` (geometry, pure numpy).

## 0. Rules
- No logos, brand names or markings. Labels are blank woven tags.
- Reference photos (the user's or the web's) drive LAYOUT and COLOUR only. Never project or paint photo pixels into a
  shipped texture (licensing + the user's private-refs rule). Detail is generated.
- Textures 2K max per item; Normal is OpenGL (flip green in UE); ORM = AO / Roughness / Metallic.
- Geometry carries silhouette only (panel outlines, thickness, puff, bends, big pouches). Everything under ~5 mm
  (stitches, webbing ribs, tacks, loop pile, wrinkles) lives in the textures.

## 1. The idea
Soft gear is flat fabric panels sewn together and bent around a body. So every item is a list of PANELS:
- `outline`: polygon in panel mm, u along the panel (and along its path), v across.
- `path` + `across` + `center`: where the panel lives in 3-D. The panel's u runs along the path (an ellipse arc round the
  torso, a spline over the shoulder, a loop for a handle), v along `across`, thickness outward away from `center`.
- `t`, `puff`, `puff_w`: thickness, pillow height, pillow roll-off (geometry AND texture use them).
- `binding`: edge-tape width (adds the tape + its stitch row). `color`: hex per panel (skins = swap colours).
- `features`: rows of `{kind: ..., ...}`, all in panel mm:

| kind | args | makes |
|---|---|---|
| `molle` | rect, rows, pitch 38.1, web 25.4 | PALS rows: 1 in webbing, 1 in gaps, bar-tacks every 1.5 in, edge stitch rows |
| `loop` | rect | hook-and-loop field: raised fuzzy pile + perimeter stitch |
| `stitch` | pts, closed | lock-stitch row (dashes, needle holes, pucker) |
| `seam` | pts, depth, width | sewn seam / fold valley + stitch row |
| `pocket` | rect, flap | sewn-on flat pocket with flap line |
| `wrinkle` | rect, angle, amp, wave | long compression / tension folds fading out of the rect |
| `label` | rect | blank tag |

`core.compose(panels)` rasterises each panel (height, material id, thread mask) at the largest texel density that fits
the atlas (2x supersampled), packs them, bleeds the gutters, adds per-material weave (Cordura, webbing twill, elastic),
then derives Normal (gradients only within a panel), AO (cavity), Roughness (per material + grime), BaseColor
(panel colour x material factor x grime x AO). `panel.shell(panel)` makes the low-poly: Delaunay of boundary + interior
grid, puffed outer face, flat inner face, side walls, all swept along the path. Outer faces UV to the panel's atlas
rect; inner faces and walls UV into one plain "inner" tile.

## 2. Make a new item (target: 15-30 min of layout, 2 min of build)
1. Copy `props/gear/plate_carrier/` to `props/gear/<item>/`; keep build.py and render.py, rewrite `spec.py`.
2. From photos, write down per panel: size (mm), shape, where on the body, features with positions. Real sizes:
   PALS 25.4 webbing / 25.4 gap / 38.1 tack pitch; SAPI medium plate 241 x 318; mag pouch 80 x 150 x 30.
3. `.venv/bin/python props/gear/<item>/build.py renders=0` -> look at the atlas (BaseColor + Normal) first: it shows
   every feature flat and is the fastest check. Then `renders=1` for the showcase shots.
4. Pouches with real volume (mag pouches, admin, GP) = box panels: front/sides/top as separate panels on straight paths,
   or reuse `props/containers/*` items placed on the PALS grid.

## 3. Fast path / batch
- Variation as data: colourways, PALS counts, cut shapes are spec numbers. A loop over specs builds a whole gear line.
- Preview at `size=1024 ss=1` (about 8x faster) while laying out; ship at `size=2048 ss=2`.
- Shared detail: for higher close-up fidelity in UE, add one tiling 512 px Cordura weave detail-normal in the master
  material (Tarkov does this) instead of raising the atlas size.
- Rigging for the character: the shell is built around the UE5 mannequin proxy in render.py; skin with a
  data-transfer of the mannequin weights (same approach as the hoodie) once the layout is approved.

## 4. Traps seen
- Wrinkle noise with a short correlation length reads as worms: use warp sigma ~45 mm and wave 10-18 mm.
- Grime noise at full strength looks blotchy on flat fabric: keep its weight low (0.15) and drive dirt from cavity.
- Panel paths must be smooth (catmull/ellipse) or the bent shell faceted; grid 25-30 mm is enough for torso bends.
- The cummerbund ellipse must sit inside the plate bags' (A-10, B-8) so it tucks under the front bag.
