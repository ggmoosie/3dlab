"""S9 gate: re-import every FBX of an asset into an empty scene, list what came back, re-run mesh_qc on it.
    .venv/bin/python tools/fbx_check.py <fbx_dir> [--expect Name=WxDxH_cm,...]   exit 0 = PASS"""
import os, sys, glob, json, runpy
import bpy
d = sys.argv[1]
exp = {}
if "--expect" in sys.argv:
    for kv in sys.argv[sys.argv.index("--expect") + 1].split(","):
        k, v = kv.split("=")
        exp[k] = [float(x) for x in v.split("x")]
ok = True
for f in sorted(glob.glob(os.path.join(d, "*.fbx"))):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=f)
    for o in sorted(bpy.data.objects, key=lambda o: o.name):
        dims = [round(x * 100, 2) for x in o.dimensions]
        tri = sum(len(p.vertices) - 2 for p in o.data.polygons) if o.type == "MESH" else 0
        line = f"{os.path.basename(f)}: {o.name} dims_cm={dims} tris={tri} mats={[s.material.name for s in o.material_slots if s.material]}"
        if o.name in exp:
            good = all(abs(a - b) <= max(0.05, 0.02 * b) for a, b in zip(sorted(dims), sorted(exp[o.name])))
            ok &= good
            line += " OK" if good else f" EXPECTED {exp[o.name]}"
        print("FBX", line)
        if o.name.startswith("UCX_"):
            o.hide_set(True)
    sys.argv = ["blender", "--", "--per-object-float"]
    try:
        runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_qc.py"), run_name="__main__")
    except SystemExit as e:
        if e.code:
            ok = False
            print("FBX QC FAIL", os.path.basename(f))
print("FBX CHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
