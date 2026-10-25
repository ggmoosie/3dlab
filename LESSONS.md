# 3dlab lessons (date: symptom → cause → fix → rule)

- 2026-10-25: water bottle labels missing → material indices set before the mesh had slots were clamped to 0; later `materials.clear()` reset them again → create slots first, assign into existing slots (`me.materials[i] = m`), never clear → treat material slots as part of the mesh contract.
- 2026-10-25: food can "pinched" spoon neck → width function discontinuous where neck meets bowl → outline = max(neck, bowl) curves → build outlines as unions of smooth curves, never piecewise jumps.
- 2026-10-25: tins rendered bare silver → curvature from pointiness on a LOW mesh is noise on big flat faces → bake CURV from a subdivided copy → pointiness needs density.
- 2026-10-25: lighter UV overlap 18 % with no duplicate faces → concave n-gons left by booleans; QC fan-triangulates them and the fans fold → triangulate (beauty) after booleans, before unwrapping.
- 2026-10-25: FBX re-import flagged decal holes → `use_custom_props=False` dropped `qc_open` → export custom props (UE ignores them).
- 2026-10-25: background build died after save → `&` inside a run_in_background command → let the tool background it; never add `&`.
- 2026-10-25: dieline prints mirrored → u must follow the viewer's right seen from outside: flip +Y, -X, -Z faces.
