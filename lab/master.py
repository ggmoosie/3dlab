"""The universal prop material (bpy). Mirrors ue/lab_master.py (M_Lab_Prop) one to one, so Blender previews = UE.

Per mesh (shared by all variants):
  T_<Item>_BaseColor  sRGB   substrate albedo (bare metal / plastic / paper) incl. baked dirt
  T_<Item>_Normal     linear OpenGL tangent normal (flip green on UE import)
  T_<Item>_ORM        linear R=AO  G=substrate roughness  B=substrate metallic
  T_<Item>_Mask       linear R=ColorA region  G=ColorB region  B=print/label region  A=wear (paint rubbed off)
Per variant (a material instance), three layers over the substrate, each with its own surface:
  Layer A (Mask.R): ColorA, RoughA, MetalA      e.g. a tinted tab, a painted cap
  Layer B (Mask.G): ColorB, RoughB, MetalB      e.g. a lid finish, a second paint
  Print   (Mask.B * Label.a): T_<Item>_Label_<Variant> (sRGB RGBA), LabelRough, non-metal
  WearAmount 0..1 rubs every layer back to the substrate where Mask.A is high; Dirt 0..1 darkens cavities (1-AO).

  for layer (c, r, m, f) in [(ColorA,RoughA,MetalA,R), (ColorB,RoughB,MetalB,G), (Label.rgb,LabelRough,0,B*Label.a)]:
      color = lerp(color, c, f); rough = lerp(rough, r, f); metal = lerp(metal, m, f)       (start = substrate)
  worn  = saturate((A - (1 - WearAmount)) * 4);  color/rough/metal = lerp(.., substrate.., worn)
  cav   = Dirt * (1 - AO);  color *= lerp(1, 0.5, cav);  rough = lerp(rough, 0.95, 0.6 * cav)
"""
import bpy, os

GROUP = "LabMaster"


def _img(path, colorspace):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = colorspace
    return im


PARAMS = (("ColorA", "C", (0.5, 0.5, 0.5)), ("RoughA", "F", 0.4), ("MetalA", "F", 0.0),
          ("ColorB", "C", (0.5, 0.5, 0.5)), ("RoughB", "F", 0.4), ("MetalB", "F", 0.0),
          ("LabelRough", "F", 0.35), ("WearAmount", "F", 0.5), ("Dirt", "F", 0.3))


def _group():
    g = bpy.data.node_groups.get(GROUP)
    if g:
        return g
    g = bpy.data.node_groups.new(GROUP, "ShaderNodeTree")
    io = g.interface
    ins = [("Substrate", "C"), ("AO", "F"), ("SubRough", "F"), ("SubMetal", "F"), ("MaskR", "F"), ("MaskG", "F"),
           ("MaskB", "F"), ("MaskA", "F"), ("Label", "C"), ("LabelA", "F")] + [(n, t) for n, t, _ in PARAMS]
    for n, t in ins:
        io.new_socket(n, in_out="INPUT", socket_type="NodeSocketColor" if t == "C" else "NodeSocketFloat")
    for n, t in [("Color", "NodeSocketColor"), ("Roughness", "NodeSocketFloat"), ("Metallic", "NodeSocketFloat")]:
        io.new_socket(n, in_out="OUTPUT", socket_type=t)
    N, L = g.nodes, g.links
    i, o = N.new("NodeGroupInput"), N.new("NodeGroupOutput")

    def math(op, a, b=None, clamp=False):
        m = N.new("ShaderNodeMath")
        m.operation, m.use_clamp = op, clamp
        for k, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[k].default_value = v
            else:
                L.new(v, m.inputs[k])
        return m.outputs[0]

    def mix(a, b, f):
        m = N.new("ShaderNodeMix")
        m.data_type = "RGBA"
        for sock, v in ((m.inputs[6], a), (m.inputs[7], b), (m.inputs[0], f)):
            if isinstance(v, (int, float, tuple)):
                sock.default_value = v
            else:
                L.new(v, sock)
        return m.outputs[2]

    def lerp(a, b, f):
        return math("ADD", a, math("MULTIPLY", math("SUBTRACT", b, a), f))

    I = i.outputs
    lab = math("MULTIPLY", I["MaskB"], I["LabelA"])
    col, rough, met = I["Substrate"], I["SubRough"], I["SubMetal"]
    for c, r, m, f in ((I["ColorA"], I["RoughA"], I["MetalA"], I["MaskR"]),
                       (I["ColorB"], I["RoughB"], I["MetalB"], I["MaskG"]),
                       (I["Label"], I["LabelRough"], 0.0, lab)):
        col, rough, met = mix(col, c, f), lerp(rough, r, f), lerp(met, m, f)
    worn = math("MULTIPLY", math("SUBTRACT", I["MaskA"], math("SUBTRACT", 1.0, I["WearAmount"])), 4.0, True)
    col = mix(col, I["Substrate"], worn)
    rough, met = lerp(rough, I["SubRough"], worn), lerp(met, I["SubMetal"], worn)
    cav = math("MULTIPLY", I["Dirt"], math("SUBTRACT", 1.0, I["AO"]))
    dark = mix((1, 1, 1, 1), (0.5, 0.5, 0.5, 1), cav)
    cm = N.new("ShaderNodeMix")
    cm.data_type, cm.blend_type = "RGBA", "MULTIPLY"
    cm.inputs[0].default_value = 1.0
    L.new(col, cm.inputs[6]); L.new(dark, cm.inputs[7])
    L.new(cm.outputs[2], o.inputs["Color"])
    L.new(lerp(rough, 0.95, math("MULTIPLY", cav, 0.6)), o.inputs["Roughness"])
    L.new(met, o.inputs["Metallic"])
    return g


def material(name, tex, params, label=None, uv="UVMap"):
    """tex: dict with BaseColor/Normal/ORM/Mask paths. params: see PARAMS (linear colours). label: RGBA print or None."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    N, L = m.node_tree.nodes, m.node_tree.links
    N.clear()
    out, bsdf = N.new("ShaderNodeOutputMaterial"), N.new("ShaderNodeBsdfPrincipled")
    grp = N.new("ShaderNodeGroup")
    grp.node_tree = _group()
    uvn = N.new("ShaderNodeUVMap")
    uvn.uv_map = uv

    def tnode(path, cs):
        t = N.new("ShaderNodeTexImage")
        t.image = _img(path, cs)
        t.interpolation = "Cubic" if cs == "sRGB" else "Linear"
        L.new(uvn.outputs[0], t.inputs[0])
        return t

    bc, orm, mk = tnode(tex["BaseColor"], "sRGB"), tnode(tex["ORM"], "Non-Color"), tnode(tex["Mask"], "Non-Color")
    sep_o, sep_m = N.new("ShaderNodeSeparateColor"), N.new("ShaderNodeSeparateColor")
    L.new(orm.outputs[0], sep_o.inputs[0]); L.new(mk.outputs[0], sep_m.inputs[0])
    L.new(bc.outputs[0], grp.inputs["Substrate"])
    for k, s in zip(("AO", "SubRough", "SubMetal"), sep_o.outputs):
        L.new(s, grp.inputs[k])
    for k, s in zip(("MaskR", "MaskG", "MaskB"), sep_m.outputs):
        L.new(s, grp.inputs[k])
    L.new(mk.outputs[1], grp.inputs["MaskA"])
    if label:
        lt = tnode(label, "sRGB")
        L.new(lt.outputs[0], grp.inputs["Label"]); L.new(lt.outputs[1], grp.inputs["LabelA"])
    else:
        grp.inputs["LabelA"].default_value = 0.0
    for k, t, d in PARAMS:
        v = params.get(k, d)
        grp.inputs[k].default_value = (*v, 1) if t == "C" else v
    nm = tnode(tex["Normal"], "Non-Color")
    nmap = N.new("ShaderNodeNormalMap")
    nmap.uv_map = uv
    L.new(nm.outputs[0], nmap.inputs["Color"])
    L.new(grp.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(grp.outputs["Roughness"], bsdf.inputs["Roughness"])
    L.new(grp.outputs["Metallic"], bsdf.inputs["Metallic"])
    L.new(nmap.outputs[0], bsdf.inputs["Normal"])
    L.new(bsdf.outputs[0], out.inputs[0])
    return m


def glass(name, tint=(0.9, 0.95, 0.92), rough=0.03, ior=1.5, normal=None):
    """Clear glass / PET. UE: M_Lab_Glass (translucent, surface-forward)."""
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*tint, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["IOR"].default_value = ior
    b.inputs["Transmission Weight"].default_value = 1.0
    if normal:
        N, L = m.node_tree.nodes, m.node_tree.links
        t = N.new("ShaderNodeTexImage")
        t.image = _img(normal, "Non-Color")
        nm = N.new("ShaderNodeNormalMap")
        L.new(t.outputs[0], nm.inputs["Color"]); L.new(nm.outputs[0], b.inputs["Normal"])
    return m


def simple(name, color, rough=0.5, metal=0.0, emission=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emission:
        b.inputs["Emission Color"].default_value = (*color, 1)
        b.inputs["Emission Strength"].default_value = emission
    return m
