# 3dlab lessons (date: symptom → cause → fix → rule)

- 2026-10-25: water bottle labels missing → material indices set before the mesh had slots were clamped to 0; later `materials.clear()` reset them again → create slots first, assign into existing slots (`me.materials[i] = m`), never clear → treat material slots as part of the mesh contract.
- 2026-10-25: food can "pinched" spoon neck → width function discontinuous where neck meets bowl → outline = max(neck, bowl) curves → build outlines as unions of smooth curves, never piecewise jumps.
- 2026-10-25: tins rendered bare silver → curvature from pointiness on a LOW mesh is noise on big flat faces → bake CURV from a subdivided copy → pointiness needs density.
- 2026-10-25: lighter UV overlap 18 % with no duplicate faces → concave n-gons left by booleans; QC fan-triangulates them and the fans fold → triangulate (beauty) after booleans, before unwrapping.
- 2026-10-25: FBX re-import flagged decal holes → `use_custom_props=False` dropped `qc_open` → export custom props (UE ignores them).
- 2026-10-25: background build died after save → `&` inside a run_in_background command → let the tool background it; never add `&`.
- 2026-10-25: dieline prints mirrored → u must follow the viewer's right seen from outside: flip +Y, -X, -Z faces.
- 2026-10-25: ammo-can latch union silently lost the plate (open edges + a "floating" tab) → `geo.rounded_box` corner radius r = 3 mm on a 3 mm-thick plate degenerates the bevel, and EXACT booleans on it drop the operand → keep r < half the smallest side; check bounds/open edges after each union when a part goes missing.
- 2026-10-25: hoodie proxy skinned only to arms/neck/thigh_twist → bpy ARMATURE_AUTO (bone heat) silently dropped pelvis/spine/thigh on a marching-cubes proxy (no warning) → solve bone heat yourself (nearest visible segment, (L+H)w=Hp with scipy splu) → always print per-bone weight totals after auto weights.
- 2026-10-25: hoodie normal bake had inverted blobs (fold peaks above the cage, facing hood-lining layers closer than the cage, cords over cloth) → bake per part group (cloth / pocket / trims) at several cage extrusions and keep, per texel, the first result with tangent z > 0.55 → layered cloth needs per-texel cage selection, not one extrusion.
- 2026-10-25: pouch weave/stitches blurred or aliased into diagonal stripes → 0.6 mm weave and 0.4 mm thread sit at Nyquist at 4.5 texels/mm (2K) → synthesise texture detail at 2x from upsampled bakes and box-filter down; pixel-align periodic weaves to an integer texel period.
