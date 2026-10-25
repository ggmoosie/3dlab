"""12 gauge shells (buck, bird: star crimp; slug: roll crimp): detailed + low meshes, bakes, variants, FBX, manifest, hero render.
    .venv/bin/python props/ammo/shell_12ga/shell_12ga.py [--fast]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rounds_lib

if __name__ == "__main__":
    rounds_lib.build("12ga", HERE)
