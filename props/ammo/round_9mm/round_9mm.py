"""9x19 mm rounds (FMJ / AP, HP): detailed + low meshes, bakes, variants, FBX, manifest, hero render.
    .venv/bin/python props/ammo/round_9mm/round_9mm.py [--fast]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import rounds_lib

if __name__ == "__main__":
    rounds_lib.build("9mm", HERE)
