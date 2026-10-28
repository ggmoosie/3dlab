---
name: photo-sdf-gun
description: Build a modular, game-ready gun high-poly from a few private reference photos with signed-distance fields (SDF) in Python, gated by photo-locked silhouette IoU. Covers profile tracing, aligning two side photos, the matched-camera compare loop, region sheets, the feature table, the slot/socket module registry (AR-15 style, aftermarket-ready), hollow parts, exploded and photo-projection renders, the traps hit on the AK-74M (each cost a rebuild), and the fast path for the next gun. Use it before starting any new gun model or when continuing guns/ak74m.
---

# Photo -> SDF -> modular gun (the AK-74M build, distilled)

Reference build: `guns/ak74m/` in ggmoosie/3dlab. Result: 27 parts, 18 slots, silhouette IoU L 0.975 / R 0.966 against the
user's two profile photos, hollow receiver + dust cover, full BCG, compensator, handguards, FCG, furniture.
Wall time on 4 cores / 15 GB: full build 3m50s, compare 1-2 min, close-up render ~1 min.

## 0. Rules that never bend
- Reference photos are PRIVATE: `refs_user/` is gitignored; any sheet/render containing photo pixels stays in `work/`
  (gitignored). Never commit, publish or bake them into a shipped texture.
- No logos, brands or markings (blank ticks on sight leaves, no stamps).
- Things that come off the gun are MODELLED (selector, charging handle, pins, levers, cover, BCG). Things that would spam
  tris are TEXTURED/BAKED (checkering, mag ribs, threads, stipple). Game textures 2K max.
- One heavy job at a time (OOM). Kill exact PIDs only.

## 1. Pipeline (files in guns/ak74m)
| Step | File | What it does | Gate |
|---|---|---|---|
| S0 contract | `CONTRACT.json`, `ak_spec.py` | variant lock, OAL/barrel/sight radius, half-widths, counted features | variant matches every photo |
| S1 profile master | `ak_ref.py` | near-ortho side photo -> mask -> per-part partition polygons -> vectorised outline -> 2D distance field `prof(name)` | trace review sheet |
| S1b second side | `align_refs.py` | mirror the other side photo, fit similarity + 2 perspective terms by mask IoU | IoU >= 0.95 (got 0.973) |
| S2 blockout | `mod_*.py`, `parts.py` | SDF per part: traced outline x real cross-section (lathe / slab / D-section / strip) | IoU >= 0.95 both sides |
| compare | `compare.py` | ortho camera locked to the photo px<->mm map, Cycles alpha, fitted camera (grid init -> Nelder-Mead 4 -> 6 params) | prints IoU, writes diff |
| regions | `regions.py` | per-region tiles (L photo, aligned R photo, model, diff) + region IoU, front to back | worst region first |
| S3 detail | `details.py` | FEATURES table: groove / pocket / boss / rivet / dimple / ring on the part's own field, by photo px | close-up renders |
| modular | `iface.py` | SLOTS (parent, socket, removable, envelope), SOCKETS (pos, axis), Y/Z landmarks | registry asserts |
| renders | `render_close.py`, `explode.py`, `projpaint.py` | studio close-ups, exploded view, PRIVATE photo projection | progress pics |
| hand-off | `npz_to_blend.py` | high.npz -> .blend/.glb (numpy+bpy only, runs in Blender's python) | opens in Blender |

Commands (repo root, `.venv` has numpy/scipy/scikit-image/pillow/bpy):
```
.venv/bin/python guns/ak74m/parts.py [parts=receiver,dcover]        # build (only listed parts; others kept)
.venv/bin/python guns/ak74m/compare.py side=LR tag=m3 row_r=287       # IoU gate
.venv/bin/python guns/ak74m/regions.py tag=m3                         # region sheet + REGION lines
.venv/bin/python guns/ak74m/render_close.py focus=receiver views="-115,30,85" hide=dcover out=work/c
.venv/bin/python guns/ak74m/explode.py views="70,18,70;-110,18,70" out=work/ex
blender -b -P guns/ak74m/npz_to_blend.py -- out=AK74M_high.blend ratio=1.0
```

## 2. How the geometry works
- Frame B (mm): bore axis z=0, muzzle face y=0, gun at y<0, +x = gun left. Blender = (x, -y, z)/1000. UE = Blender (x, -y, z).
- Outline: `R.prof(name)(Y, Z)` = signed distance of the photo silhouette inside that part's partition polygon. Width comes
  from `slab(P2, X, hw, r)` or a lathe (`lathe_photo`: radius per photo column) or a custom section (dust cover D-top).
- Parametric where tracing fails: thin light parts (trigger guard strip along a measured centre line), levers
  (`chain2` capsules along photo points), anything hidden (barrel under handguards, BCG, trunnions) from `iface` landmarks.
- Detail: `details.FEATURES[part]` rows in PHOTO PX. Kinds operate on the part's own distance field so a rib follows any
  surface. Variation as data: adding a rivet = adding a row.
- Hollow parts: `PARTS` row has `outer` + `inner` (cavity field). Build = details on OUTER, then `max(d, -inner)`.
  Open-top trick: evaluate the outline at `min(Z, z_cut)` so the cavity runs out through the top.
- Mesh: narrow-band marching cubes per block (`lab/sdf.py`), h 0.2-0.5 mm; thin walls need h <= wall/3.
- Module registry: each `mod_*.py` exports `PARTS = [dict(name, slot, mat, h, outer, inner?, trace|box, xh?)]`;
  `parts.py` asserts unique names + known slots and writes `review/modules.json` (slots, sockets, parts).
  A Zenit B-10/B-11/B-33 class part = a new mod file filling an existing slot inside its `envelope`.

## 3. Traps (each cost at least one rebuild) -> rule
| Trap | Symptom | Rule |
|---|---|---|
| camera `to_track_quat("-Z","Z")` | render 90 deg rolled, IoU ~0.3 | always `("-Z","Y")` |
| Field2D passed, not evaluated | numpy error / empty part | `R.prof(n)(Y, Z)` |
| other side photo not registered | R IoU 0.855, white seams in projection | fit the camera per side; apply it everywhere (compare, projection) |
| shiny pin heads / specular streaks read as background | holes in parts, wavy slot along the dust cover | fill small holes (<90 px) and hair-thin long holes (aspect > 25) in the mask |
| partition polygons leave gaps or overlap | red bars in the diff, double-traced bumps | partitions share edges; check every red bar against the polygon table |
| traced thin light parts | broken trigger guard | thin/light parts are parametric strips along measured points |
| pixel stairs | faceted bevels | sub-pixel contours, arc smoothing, Douglas-Peucker, smoothed distance field |
| polygon SDF orientation | features shrink the whole part instead of cutting a band | unit-test `_poly2` sign on an inside point before building |
| inequality sign on a trim (`cap - Y` vs `Y - cap`) | whole part vanished | smoke-test inside-fraction of every part before meshing (see 4) |
| pocket depth >= wall | punched holes in a 1 mm shell | details before hollowing; depth <= 0.5 x wall |
| big edge rounding on a shell | see-through slit between handguards | cavity = true offset of the OUTER field (+ a top skin), not the shrunk outline |
| part pokes into a neighbour by sign slip | carrier into rear sight block | interfaces come from `iface.Y/Z`, never retyped numbers |

## 4. Fast path for the next gun (do these from day one)
1. **Modular first.** Write `iface.py` (landmarks, slots, sockets, envelopes) and one `mod_*.py` per slot BEFORE any
   detail. The AK was built solid, then refactored: about 2 hours wasted.
2. **Feature census before modelling.** One pass over every photo -> `details.FEATURES` rows + list of removable parts
   (charging handle, selector, pins, levers). The AK missed its charging handle and ejection port until late.
3. **Smoke test before meshing (seconds, not minutes).** For every part: sample 2000 random points in its bbox, print the
   inside fraction; 0 or ~1 means a sign bug. Also unit-test any new 2D helper on one inside point.
4. **Preview at `h_scale=2`** (8x fewer voxels) while iterating; full res only for the gate and the hand-off.
5. **Build only what changed:** `parts=` list now; next step is a per-part hash cache (source of its mod file + spec +
   photos), same idea as the SCAR bake cache.
6. **Gate per region, not globally.** `regions.py` REGION lines after every build; fix the worst region first.
7. **Projection check early.** `projpaint.py` on the blockout shows misplaced features in one render (it is private).
8. **Exploded render script ready from S2** so every module is checked in context.
9. **Parallelise by module.** Mod files are independent: one agent per mod file with `iface.py` frozen; the main thread
   only runs compare/regions. Expect 3-4x on the S3 pass.
10. **Do not hunt blueprints.** Manufacturing drawings are off limits and unnecessary: profile photo + overall length
    calibrate to ~1 mm. Ask the user for field-strip photos, a top view and a front view instead.
11. **One review round per gate, top 3 issues** (gun-pipeline rule). 90-95 % then move on.

## 5. Open items on the AK-74M (next agent starts here)
- Rear sight: split base / leaf / leaf spring (now one traced block + slider).
- Front sight: ears + post shapes; FSB detent for the brake.
- Dust cover stiffening ribs; check charging handle against 3/4 photos 163300/163304.
- Carrier underside groove + piston pin; FCG auto sear, hammer retarder, disconnector.
- Mag follower + spring; mag ribs go to the bake.
- Stock hinge block, latch, buttplate trapdoor.
- Game kit (AR-15 pattern): SM_AK74M_{Receiver, DustCover, BCG, Recoil, Barrel, Muzzle_74, HG_Upper, HG_Lower, RearSight,
  Grip, Stock, Mag30, Mag30_Loaded}, each with LOD1/LOD2, UCX, sockets JSON, 2K textures; `game_parts.json` generated from
  `iface.SLOTS`; loose `parts/` kit for exploded views. Reuse the SCAR low-poly/bake-cache/export scripts.
