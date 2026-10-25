"""UE 5.5 editor python: build M_Lab_Prop (the universal mask master), mirroring lab/master.py exactly.
Run once (Tools > Execute Python Script, or py "ue/lab_master.py"). Idempotent: skips if the .uasset exists unless
REBUILD=True. Lessons applied (model-factory §5/§6): recompile_material after edits, save the master before any
MI is made, never create_asset over an existing path from an unattended script."""
import os
import unreal

PATH, NAME = "/Game/Lab/Materials", "M_Lab_Prop"
REBUILD = False
mel = unreal.MaterialEditingLibrary
at = unreal.AssetToolsHelpers.get_asset_tools()


def _exists(path):
    rel = path.replace("/Game/", "", 1) + ".uasset"
    return os.path.exists(os.path.join(unreal.Paths.project_content_dir(), rel))


class Graph:
    def __init__(self, mat):
        self.m, self.x = mat, -1600

    def node(self, cls, y=0, **props):
        e = mel.create_material_expression(self.m, cls, self.x, y)
        for k, v in props.items():
            e.set_editor_property(k, v)
        self.x += 0 if props.get("_keep_x") else 0
        return e

    def link(self, a, a_out, b, b_in):
        mel.connect_material_expressions(a, a_out, b, b_in)

    def const(self, v, y=0):
        return self.node(unreal.MaterialExpressionConstant, y, r=v)

    def op(self, cls, a, b=None, a_out="", b_out="", y=0):
        e = self.node(cls, y)
        self.link(a, a_out, e, "A" if b is not None or cls is not unreal.MaterialExpressionOneMinus else "")
        if b is not None:
            self.link(b, b_out, e, "B")
        return e

    def lerp(self, a, b, alpha, a_out="", b_out="", al_out="", y=0):
        e = self.node(unreal.MaterialExpressionLinearInterpolate, y)
        self.link(a, a_out, e, "A"); self.link(b, b_out, e, "B"); self.link(alpha, al_out, e, "Alpha")
        return e


def build():
    full = f"{PATH}/{NAME}"
    if _exists(full) and not REBUILD:
        unreal.log(f"{full} exists; set REBUILD=True to rebuild it in place")
        return unreal.load_asset(f"{full}.{NAME}")
    if _exists(full):
        mat = unreal.load_asset(f"{full}.{NAME}")
        for e in [x for x in unreal.ObjectIterator(unreal.MaterialExpression) if x.get_outer() == mat]:
            mel.delete_material_expression(mat, e)
    else:
        mat = at.create_asset(NAME, PATH, unreal.Material, unreal.MaterialFactoryNew())
    g = Graph(mat)
    T = unreal.MaterialExpressionTextureSampleParameter2D
    def tex(name, sampler, default, y):
        return g.node(T, y, parameter_name=name, sampler_type=sampler, texture=unreal.load_asset(default))
    ST = unreal.MaterialSamplerType
    D = "/Game/Lab/Textures/Defaults"                     # imported by lab_import.defaults() first
    base = tex("BaseColor", ST.SAMPLERTYPE_COLOR, f"{D}/T_Lab_Grey", -600)
    nrm = tex("Normal", ST.SAMPLERTYPE_NORMAL, "/Engine/EngineMaterials/DefaultNormal", -300)
    orm = tex("ORM", ST.SAMPLERTYPE_MASKS, f"{D}/T_Lab_White", 0)
    msk = tex("Mask", ST.SAMPLERTYPE_MASKS, f"{D}/T_Lab_Black", 300)
    lab = tex("Label", ST.SAMPLERTYPE_COLOR, f"{D}/T_Lab_Clear", 600)
    V, S = unreal.MaterialExpressionVectorParameter, unreal.MaterialExpressionScalarParameter
    def vec(n, c, y):
        return g.node(V, y, parameter_name=n, default_value=unreal.LinearColor(r=c[0], g=c[1], b=c[2], a=1.0))
    def sca(n, v, y):
        return g.node(S, y, parameter_name=n, default_value=v)
    g.x = -1200
    colA, roughA, metA = vec("ColorA", (0.5, 0.5, 0.5), 900), sca("RoughA", 0.4, 1000), sca("MetalA", 0.0, 1100)
    colB, roughB, metB = vec("ColorB", (0.5, 0.5, 0.5), 1200), sca("RoughB", 0.4, 1300), sca("MetalB", 0.0, 1400)
    labR, useL = sca("LabelRough", 0.35, 1500), sca("UseLabel", 0.0, 1600)
    wearAmt, dirt = sca("WearAmount", 0.5, 1700), sca("Dirt", 0.3, 1800)
    M = unreal.MaterialExpressionMultiply
    g.x = -800
    labA = g.op(M, g.op(M, msk, lab, "B", "A"), useL)                         # print coverage
    col, rough, met = base, orm, orm
    c_out, r_out, m_out = "RGB", "G", "B"
    for c, r, m, f, f_out in ((colA, roughA, metA, msk, "R"), (colB, roughB, metB, msk, "G"), (lab, labR, None, labA, "")):
        col = g.lerp(col, c, f, c_out, "RGB" if c is lab else "", f_out); c_out = ""
        rough = g.lerp(rough, r, f, r_out, "", f_out); r_out = ""
        met = g.lerp(met, m if m else g.const(0.0), f, m_out, "", f_out); m_out = ""
        g.x += 120
    # worn = saturate((A - (1 - WearAmount)) * 4)
    worn = g.node(unreal.MaterialExpressionSaturate)
    w0 = g.op(M, g.op(unreal.MaterialExpressionSubtract, msk, g.op(unreal.MaterialExpressionOneMinus, wearAmt), "A"), g.const(4.0))
    g.link(w0, "", worn, "")
    col = g.lerp(col, base, worn, "", "RGB")
    rough = g.lerp(rough, orm, worn, "", "G")
    met = g.lerp(met, orm, worn, "", "B")
    cav = g.op(M, dirt, g.op(unreal.MaterialExpressionOneMinus, orm, a_out="R"))
    col = g.op(M, col, g.lerp(g.const(1.0), g.const(0.5), cav))
    rough = g.lerp(rough, g.const(0.95), g.op(M, cav, g.const(0.6)))
    MP = unreal.MaterialProperty
    mel.connect_material_property(col, "", MP.MP_BASE_COLOR)
    mel.connect_material_property(rough, "", MP.MP_ROUGHNESS)
    mel.connect_material_property(met, "", MP.MP_METALLIC)
    mel.connect_material_property(orm, "R", MP.MP_AMBIENT_OCCLUSION)
    mel.connect_material_property(nrm, "RGB", MP.MP_NORMAL)
    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    unreal.log(f"built {full}")
    return mat


if __name__ == "__main__":
    build()
