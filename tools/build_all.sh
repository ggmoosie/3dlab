#!/bin/bash
# Full build + QC + FBX re-import gate for each prop script given. Logs to work/build_<name>.log; prints a verdict line each.
cd "$(dirname "$0")/.."
for script in "$@"; do
  name=$(basename "$script" .py); dir=$(dirname "$script")
  log=work/build_$name.log
  .venv/bin/python "$script" > "$log" 2>&1; b=$?
  objs=$(.venv/bin/python -c "import json;m=json.load(open('$dir/manifest.json'));print(','.join(k for k in m['tris'] if '_LOD' not in k))" 2>/dev/null)
  .venv/bin/python tools/qc.py "$dir/$name.blend" --objects "$objs" --per-object-float $QC_EXTRA >> "$log" 2>&1; q=$?
  .venv/bin/python tools/fbx_check.py "$dir/fbx" >> "$log" 2>&1; f=$?
  echo "BUILD $name build=$b qc=$q fbx=$f $(grep -E '^TRIS' $log | tail -1)"
done
