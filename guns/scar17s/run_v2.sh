#!/bin/bash
# SCAR 17S v2 full chain: high-poly -> side compare -> game meshes -> 3/4 camera + compare -> bake -> export/renders
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python; L=work
$PY guns/scar17s/parts_v2.py > $L/scar_v2_parts.log 2>&1 && grep HIGH_DONE $L/scar_v2_parts.log || exit 1
$PY guns/scar17s/compare.py src=high samples=24 tag=v3 2>&1 | grep COMPARE
$PY guns/scar17s/bl_low.py > $L/scar_low.log 2>&1; grep -E "LOW_DONE|Traceback" $L/scar_low.log
$PY -c "
import bpy,sys,runpy
bpy.ops.wm.open_mainfile(filepath='guns/scar17s/work/scar_low.blend')
for o in bpy.data.objects:
    if o.name.startswith('HI_') or o.name.endswith('Round'): o.hide_render=True
sys.argv=['blender','--','guns/scar17s/work/match_left34.json']
runpy.run_path('tools/camera_match.py', run_name='__main__')
" 2>&1 | grep -E "CAM (rms|VERDICT)"
$PY tools/ref_compare.py guns/scar17s/refs_user/nrch_left34.jpg guns/scar17s/review/render_left34.png guns/scar17s/review/compare_left34 2>&1 | grep COMPARE
(cd guns/scar17s && ../../$PY silfit.py work/match_left34.json review/cam_left34.json 2>&1 | grep SILFIT)
$PY guns/scar17s/bl_bake.py > $L/scar_bake.log 2>&1; grep -E "BAKE_DONE|Traceback" $L/scar_bake.log
$PY guns/scar17s/bl_final.py stage=all samples=64 > $L/scar_final.log 2>&1; grep -E "TRIS|Traceback" $L/scar_final.log | tail -2
$PY guns/scar17s/compare.py src=game samples=48 tag=v3 2>&1 | grep COMPARE
$PY guns/scar17s/showcase.py shots=hero34,fieldstrip samples=128 > $L/showcase.log 2>&1; grep -E "Saved: .*showcase|Traceback" $L/showcase.log
echo CHAIN_DONE
