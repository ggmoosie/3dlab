# SCAR 17S: status

**Variant (locked):** FN SCAR 17S NRCH, 7.62x51, 16.25 in barrel (1:12), FDE. Skins: FDE, Black, Camo (one mask material, three MIs per receiver part).

## Pipeline (all scripts, rerunnable)
| Stage | Script | Output |
|---|---|---|
| Spec | `scar_spec.py` | dimensions in the B frame (bore z 0, bolt face y 0); game frame G = origin at the grip top |
| High-poly (S2/S3) | `parts.py` (SDF + narrow-band marching cubes, `lab/sdf.py` from the Glock v4) | 18 parts, 12.2 M tris, `work/high.npz` (about 2 min) |
| Game mesh (S7) | `bl_low.py` (Glock v4 recipe: collapse, sliver fix, weighted normals, packed UVs) | 10 meshes, 47.5 k tris LOD0, `work/scar_low.blend` |
| QC (S4) | `tools/qc.py` | **0 errors**; warnings = long thin triangles on lathe surfaces (barrel, stock edges), justified |
| Bake (S8) | `bl_bake.py` | per mesh NORMAL / AO / CURV / ID / POS -> BaseColor, Normal, ORM, Mask (+ camo label from 3-D noise) |
| Final (S9) | `bl_final.py` | FBX (+LOD1 45 % / LOD2 18 %, UCX hulls), merged ItemMesh `SM_SCAR17S`, sockets + mechanism json, renders, exploded sheet |

## Game meshes (LOD0 tris)
Upper 15.3k (receiver, rails 3/6/9/12, hinge block, latch, folding sights) · Lower 8.8k (trigger, ambi safety + mag release, bolt catch, pins) ·
Stock 7.4k (body, cheek riser, buttpad) · Mag 4.0k · Barrel 3.7k (with gas block + regulator) · Bolt 2.6k · Grip 2.2k · Muzzle 2.0k ·
ChargingHandle 1.1k · Round 0.5k. Textures: Upper 4K, Lower / Stock / Barrel / Mag 2K, rest 1K.

## Moving parts (sockets/SM_SCAR17S_mechanism.json)
Bolt (travel 11.8 cm), charging handle (NRCH, 25.5 cm slot, does not reciprocate), stock (folds right about the hinge), magazine (at MagazineSocket).

## Open items (stated plainly)
1. **No reference photos.** This cloud box's network policy blocks image hosts, so the S2 silhouette IoU gate and the S5 spot-the-difference
   diffs could not run. Shapes come from the published specs and the well-known layout. Send 3-6 photos of a 17S NRCH (one near-orthographic
   right side) and I will run camera_match + ref_compare and fix the diff list.
2. Sockets are computed from the spec, not ray-measured on the mesh (handoff_check does that on import).
3. Shapes I'm least sure of without photos: the stock's side profile, the lower's exact outline around the magwell, the muzzle device pattern.
4. UE side is untested here (no Unreal in the cloud box).
