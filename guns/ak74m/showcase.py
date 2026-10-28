"""showcase.py - model-only showcase close-ups of the AK-74M high-poly (no photos): parkerized steel, blued steel, black
polymer with procedural micro-variation (object-space noise roughness + fine bump), grey studio sweep (lab.scene.studio).
usage: .venv/bin/python guns/ak74m/showcase.py [shots=hero,right,muzzle,receiver,bcg,stock] [spp=64] [w=1600] out=work/show"""
import bpy, os, sys, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
A = dict(a.split("=", 1) for a in sys.argv if "=" in a); sys.argv = ["x"]
import compare as C
from lab import scene

# name: (focus parts, hidden parts, yaw, pitch, lens, fill)
SHOTS = {
    "hero":     (None, (), 62, 14, 60, 0.9),
    "right":    (None, (), -118, 16, 60, 0.9),
    "muzzle":   (("compensator", "fsb", "gasblock"), (), 38, 10, 100, 0.85),
    "receiver": (("selector", "receiver", "rsight", "magcatch", "tguard"), (), -105, 12, 85, 0.85),
    "bcg":      (("carrier", "bolt", "recoil", "receiver"), ("dcover",), -125, 34, 70, 0.85),
    "stock":    (("stock", "grip", "sling_loop"), (), -70, 8, 70, 0.85),
}
LOOK = {0: ((0.030, 0.030, 0.032), 0.85, 0.38, 0.6),          # mat id: base, metallic, roughness, rough variation
        1: ((0.022, 0.022, 0.023), 0.0, 0.62, 0.25),
        2: ((0.040, 0.040, 0.040), 0.7, 0.55, 0.5)}


def material(k):
    base, met, rough, var = LOOK[k]
    m = bpy.data.materials.new(f"AK_{k}"); m.use_nodes = True; N = m.node_tree.nodes; L = m.node_tree.links
    b = N["Principled BSDF"]; b.inputs["Base Color"].default_value = (*base, 1); b.inputs["Metallic"].default_value = met
    tc = N.new("ShaderNodeTexCoord")
    nz = N.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 60.0; nz.inputs["Detail"].default_value = 8.0
    L.new(tc.outputs["Object"], nz.inputs["Vector"])
    mr = N.new("ShaderNodeMapRange"); L.new(nz.outputs["Fac"], mr.inputs[0])
    mr.inputs[3].default_value = rough * (1 - var * 0.4); mr.inputs[4].default_value = min(rough * (1 + var * 0.4), 0.95)
    L.new(mr.outputs[0], b.inputs["Roughness"])
    fine = N.new("ShaderNodeTexNoise"); fine.inputs["Scale"].default_value = 900.0; L.new(tc.outputs["Object"], fine.inputs["Vector"])
    bump = N.new("ShaderNodeBump"); bump.inputs["Strength"].default_value = 0.08 if k == 1 else 0.04; bump.inputs["Distance"].default_value = 0.0002
    L.new(fine.outputs["Fac"], bump.inputs["Height"]); L.new(bump.outputs["Normal"], b.inputs["Normal"])
    return m


def main():
    C.load()
    mats = {k: material(k) for k in LOOK}
    objs = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
    for o in objs.values():
        k = int(o.data.materials[0].name[1]) if o.data.materials else 0
        o.data.materials.clear(); o.data.materials.append(mats[k])
    scene.studio(list(objs.values()), backdrop=True, key=1.0)
    out = A.get("out", os.path.join(HERE, "work", "show"))
    for name in A.get("shots", ",".join(SHOTS)).split(","):
        focus, hide, yaw, pitch, lens, fill = SHOTS[name]
        for o in objs.values():
            o.hide_render = o.name in hide
        fo = [objs[n] for n in focus] if focus else list(objs.values())
        scene.frame(fo, yaw=yaw, pitch=pitch, lens=lens, fill=fill, name=f"cam_{name}", aspect=0.625)
        w = int(A.get("w", 1600))
        scene.render(f"{out}_{name}.png", res=w, res_y=int(w * 0.625), samples=int(A.get("spp", 64)))
        print("SHOT", name)


if __name__ == "__main__":
    main()
