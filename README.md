# 3dlab

From-scratch game assets for **LootNShoot** (UE 5.5), built entirely by Python scripts in headless Blender 5
(`bpy` as a module). No downloaded models, textures or scans; no real brands. Every asset folder rebuilds itself.

```sh
uv venv -p 3.11 .venv && uv pip install -p .venv bpy numpy pillow scipy scikit-image
.venv/bin/python props/drinks/soda_can/soda_can.py            # build: mesh, bakes, textures, variants, renders, FBX
.venv/bin/python tools/qc.py props/drinks/soda_can/soda_can.blend --collection Game --per-object-float   # mesh_qc gate
.venv/bin/python tools/fbx_check.py props/drinks/soda_can/fbx  # S9: re-import + QC every exported FBX
```

## Layout
| Path | What |
|---|---|
| `lab/` | shared library: `scene` (studio, cameras, render), `geo` (lathe with planned UV islands), `tex` (tileable noise, wear, normals), `labels` (packaging design kit in cm), `bake` (high→low normal/AO/curvature/ID), `master` (universal material), `export` (FBX, UCX, LODs) |
| `tools/` | `mesh_qc.py` (the model-factory gate), `qc.py` / `fbx_check.py` wrappers, `sheet.py` contact sheets |
| `ue/` | `lab_import.py` imports any prop from its `manifest.json`; `lab_master.py` builds the `M_Lab_Prop` master |
| `props/<category>/<item>/` | `<item>.py` + helpers, `<item>.blend`, `fbx/`, `textures/`, `renders/`, `manifest.json` |
| `fonts/` | OFL fonts used for label art (licences alongside) |

## Universal mask material (`M_Lab_Prop`)
One texture set per mesh, any number of variants as material instances:
- `T_<Item>_BaseColor` substrate albedo · `T_<Item>_Normal` OpenGL · `T_<Item>_ORM` (AO, roughness, metallic)
- `T_<Item>_Mask`: **R** layer A · **G** layer B · **B** print area · **A** wear
- per variant: `T_<Item>_Label_<V>` (RGBA print) + ColorA/RoughA/MetalA, ColorB/RoughB/MetalB, LabelRough, WearAmount, Dirt

`lab/master.py` (Blender preview) and `ue/lab_master.py` (UE) implement the same math, so renders match the game.

## In Unreal
`py "<3dlab>/ue/lab_import.py" <3dlab>/props/drinks/soda_can` → `/Game/Lab/Props/SodaCan/` with meshes, UCX, LODs,
textures (normal maps green-flipped), `MI_SodaCan_<Variant>` and a verify report `ue_import.json`.
