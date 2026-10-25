"""7.62x51 mm NATO rounds (FMJ / AP): detailed + low meshes, bakes, variants, FBX, manifest, hero render.
    .venv/bin/python props/ammo/round_762/round_762.py [--fast]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rounds_lib

if __name__ == "__main__":
    rounds_lib.build("762", HERE)
