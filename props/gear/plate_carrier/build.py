"""build.py - plate carrier: spec panels -> 2K PBR atlas (texture-space modelling) + low-poly mesh -> FBX + showcase renders.
usage: .venv/bin/python props/gear/plate_carrier/build.py [size=2048] [renders=1] [spp=64]"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "../../..")
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
from lab import tex as T
from lab.softgoods import core, panel as PN
import spec

ITEM = "PlateCarrier"
TEX = os.path.join(HERE, "textures"); OUT = os.path.join(HERE, "fbx"); REN = os.path.join(HERE, "renders")
SIZE = int(A.get("size", 2048))
INNER = dict(name="inner", outline=[(0, 0), (160, 0), (160, 160), (0, 160)], color=spec.COLOR, puff_tex=0.3,
             features=[dict(kind="spacer", rect=(0, 0, 160, 160))])                   # undersides: spacer mesh


def maps():
    pans = spec.panels() + [INNER]
    m, rects, tpm = core.compose(pans, size=SIZE, ss=int(A.get("ss", 2)))
    for k, fn in (("albedo", "BaseColor"), ("normal", "Normal"), ("orm", "ORM")):
        T.save(m[k], os.path.join(TEX, f"T_{ITEM}_{fn}.png"))
    return pans[:-1], {r["name"]: r for r in rects}, tpm


def uv_of(r, uv_mm, tpm):
    x, y, w, h = r["px"]; u0, v0, u1, v1 = r["mm"]
    u = (x + (uv_mm[..., 0] - u0) * tpm) / SIZE
    v = 1 - (y + (v1 - uv_mm[..., 1]) * tpm) / SIZE
    return np.stack([u, v], -1)


def inner_uv(r, uv_mm, outline):
    O = np.asarray(outline, float); lo, hi = O.min(0), O.max(0)
    q = (uv_mm - lo) / np.maximum(hi - lo, 1e-6)
    x, y, w, h = r["px"]
    return np.stack([(x + 4 + q[..., 0] * (w - 8)) / SIZE, 1 - (y + 4 + (1 - q[..., 1]) * (h - 8)) / SIZE], -1)


def main():
    import bpy
    pans, rects, tpm = maps()
    print(f"ATLAS {SIZE}px  {tpm:.2f} texel/mm")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    allV, allF, allUV = [], [], []; off = 0
    for p in pans:
        V, F, UV, side = PN.shell(p, grid=float(A.get("grid", 30)))
        uv = np.where((side == 0)[:, None, None], uv_of(rects[p["name"]], UV, tpm), inner_uv(rects["inner"], UV, p["outline"]))
        allV.append(V); allF.append(F + off); allUV.append(uv); off += len(V)
    V = np.vstack(allV) * 0.001; F = np.vstack(allF); UV = np.vstack(allUV)
    me = bpy.data.meshes.new(f"SM_{ITEM}"); me.vertices.add(len(V)); me.vertices.foreach_set("co", V.ravel())
    me.loops.add(F.size); me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F)); me.polygons.foreach_set("loop_start", np.arange(0, F.size, 3)); me.polygons.foreach_set("loop_total", np.full(len(F), 3))
    me.update(calc_edges=True); me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    me.uv_layers.new(name="UVMap").data.foreach_set("uv", UV.reshape(-1, 2).ravel())
    ob = bpy.data.objects.new(f"SM_{ITEM}", me); bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob; ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT"); bpy.ops.mesh.remove_doubles(threshold=0.0002)
    bpy.ops.object.mode_set(mode="OBJECT")
    me.set_sharp_from_angle(angle=np.radians(50)) if hasattr(me, "set_sharp_from_angle") else None
    mat = bpy.data.materials.new(f"M_{ITEM}"); mat.use_nodes = True; N = mat.node_tree.nodes; L = mat.node_tree.links
    b = N["Principled BSDF"]
    def img(fn, cs):
        t = N.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(os.path.join(TEX, f"T_{ITEM}_{fn}.png")); t.image.colorspace_settings.name = cs; return t
    L.new(img("BaseColor", "sRGB").outputs[0], b.inputs["Base Color"])
    orm = img("ORM", "Non-Color"); sp = N.new("ShaderNodeSeparateColor"); L.new(orm.outputs[0], sp.inputs[0]); L.new(sp.outputs[1], b.inputs["Roughness"])
    nm = N.new("ShaderNodeNormalMap"); L.new(img("Normal", "Non-Color").outputs[0], nm.inputs["Color"]); L.new(nm.outputs[0], b.inputs["Normal"])
    b.inputs["Sheen Weight"].default_value = 0.3 if "Sheen Weight" in b.inputs else 0
    ob.data.materials.append(mat)
    tris = len(F); print("TRIS", tris)
    os.makedirs(OUT, exist_ok=True)
    bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, f"SM_{ITEM}.fbx"), use_selection=True, apply_unit_scale=True, mesh_smooth_type="FACE")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, f"{ITEM.lower()}.blend"))
    json.dump(dict(item=ITEM, tris=tris, atlas=SIZE, texel_per_mm=round(tpm, 3), panels=[p["name"] for p in pans],
                   textures={k: f"textures/T_{ITEM}_{k}.png" for k in ("BaseColor", "Normal", "ORM")},
                   note="Normal = OpenGL (flip green in UE). ORM = AO/Roughness/Metallic. Layout v0 (generic, no branding)."),
              open(os.path.join(HERE, "manifest.json"), "w"), indent=1)
    if A.get("renders", "1") == "1":
        import render; render.shots(ob, REN, int(A.get("spp", 64)))


if __name__ == "__main__":
    main()
