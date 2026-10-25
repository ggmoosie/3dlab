"""preview.py - quick high-poly review renders: right side, left 3/4, exploded.
usage: .venv/bin/python guns/scar17s/preview.py [samples=16] [views=side,left,exploded] [out=review/prefix]"""
import bpy, os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
from lab import scene
import bl_high

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SMP = int(ARGS.get("samples", 16))
VIEWS = ARGS.get("views", "side,left,exploded").split(",")
OUT = os.path.join(HERE, ARGS.get("out", "review/hp"))

# exploded layout (Blender metres, offsets from the assembled pose): stock back, mag/grip down, barrel forward...
EXPLODE = {"barrel": (0, -0.16, 0), "gasblock": (0, -0.20, 0.05), "hider": (0, -0.26, 0), "bolt": (0, 0.0, 0.12),
           "charging": (0.0, -0.05, 0.14), "lower": (0, 0, -0.09), "trigger": (0, 0, -0.16), "controls": (0, 0.02, -0.21),
           "pins": (0, 0.0, -0.26), "grip": (0, 0.05, -0.25), "magazine": (0, -0.05, -0.30), "round": (0, -0.05, -0.40),
           "stock": (0, 0.12, 0), "buttpad": (0, 0.18, 0), "rsight": (0, 0, 0.07), "fsight": (0, 0, 0.07), "latch": (0.05, 0, 0)}


def main():
    scene.reset()
    objs = bl_high.load()
    allo = list(objs.values())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    scene.studio(allo)
    for v in VIEWS:
        for n, o in objs.items():
            o.location = EXPLODE.get(n, (0, 0, 0)) if v == "exploded" else (0, 0, 0)
        bpy.context.view_layer.update()
        if v == "side":
            scene.frame(allo, yaw=-90, pitch=0, lens=120, fill=0.94, name="side", aspect=0.42)
            scene.render(f"{OUT}_side.png", 1800, SMP, res_y=756)
        elif v == "left":
            scene.frame(allo, yaw=60, pitch=16, lens=70, fill=0.92, name="left", aspect=0.55)
            scene.render(f"{OUT}_left.png", 1800, SMP, res_y=990)
        elif v == "exploded":
            scene.frame(allo, yaw=-90, pitch=6, lens=90, fill=0.92, name="expl", aspect=0.62)
            scene.render(f"{OUT}_exploded.png", 1800, SMP, res_y=1116)
        elif v == "front":
            scene.frame(allo, yaw=-160, pitch=12, lens=90, fill=0.8, name="front", aspect=0.75)
            scene.render(f"{OUT}_front.png", 1400, SMP, res_y=1050)


if __name__ == "__main__":
    main()
