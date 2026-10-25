"""Run mesh_qc.py (the model-factory gate) on a .blend with the bpy module. Exit code = verdict (0 pass, 1 error).
    .venv/bin/python tools/qc.py file.blend [mesh_qc options]"""
import os, sys, runpy
import bpy

blend, rest = sys.argv[1], sys.argv[2:]
bpy.ops.wm.open_mainfile(filepath=blend)
sys.argv = ["blender", "--"] + rest
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_qc.py"), run_name="__main__")
