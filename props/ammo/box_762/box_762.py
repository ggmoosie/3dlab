"""762 ammo box: tuck-top folded board box, lid (moving part), pull-out tray; label art, FBX, manifest, renders.
    .venv/bin/python props/ammo/box_762/box_762.py [--fast]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import box_lib

if __name__ == "__main__":
    box_lib.build("762", HERE)
