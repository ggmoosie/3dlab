"""UE 5.5 editor python: import 3dlab props from their manifest.json into /Game/Lab/Props/<Item>/.
    py "<3dlab>/ue/lab_import.py" <3dlab>/props/drinks/soda_can [more prop folders...]
Per prop: textures (correct compression/sRGB per role), FBX meshes + UCX + LODs, one MI per variant (M_Lab_Prop),
default MI assigned. Idempotent: existing .uasset files are reused, never overwritten (set REIMPORT=True to refresh).
Verify step writes <prop>/ue_import.json (tris per LOD, collision hulls, materials) read back from the assets."""
import os, sys, json, glob
import unreal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lab_master

REIMPORT = False
at = unreal.AssetToolsHelpers.get_asset_tools()
eal = unreal.EditorAssetLibrary
DEF = "/Game/Lab/Textures/Defaults"
TC = unreal.TextureCompressionSettings


def exists(path):
    return os.path.exists(os.path.join(unreal.Paths.project_content_dir(), path.replace("/Game/", "", 1) + ".uasset"))


def import_file(src, dest, name=None):
    name = name or os.path.splitext(os.path.basename(src))[0]
    if exists(f"{dest}/{name}") and not REIMPORT:
        return unreal.load_asset(f"{dest}/{name}.{name}")
    t = unreal.AssetImportTask()
    t.filename, t.destination_path, t.destination_name = src, dest, name
    t.automated, t.replace_existing, t.save = True, True, True
    at.import_asset_tasks([t])
    return unreal.load_asset(f"{dest}/{name}.{name}")


def texture(src, dest, role):
    """role: color | label | normal | masks"""
    tx = import_file(src, dest)
    if role == "normal":
        tx.set_editor_property("compression_settings", TC.TC_NORMALMAP)
        tx.set_editor_property("flip_green_channel", True)            # OpenGL bake → UE (DirectX)
        tx.set_editor_property("srgb", False)
    elif role == "masks":
        tx.set_editor_property("compression_settings", TC.TC_MASKS)
        tx.set_editor_property("srgb", False)
    else:
        tx.set_editor_property("compression_settings", TC.TC_DEFAULT)
        tx.set_editor_property("srgb", True)
    eal.save_loaded_asset(tx)
    return tx


def defaults():
    d = os.path.join(HERE, "defaults")
    return {"white": texture(f"{d}/T_Lab_White.png", DEF, "masks"), "black": texture(f"{d}/T_Lab_Black.png", DEF, "masks"),
            "clear": texture(f"{d}/T_Lab_Clear.png", DEF, "label"), "color": texture(f"{d}/T_Lab_Grey.png", DEF, "color")}


def import_mesh(fbx, dest):
    name = os.path.splitext(os.path.basename(fbx))[0]
    if exists(f"{dest}/{name}") and not REIMPORT:
        return unreal.load_asset(f"{dest}/{name}.{name}")
    t = unreal.AssetImportTask()
    t.filename, t.destination_path, t.destination_name = fbx, dest, name
    t.automated, t.replace_existing, t.save = True, True, True
    o = unreal.FbxImportUI()
    o.import_materials = o.import_textures = o.import_as_skeletal = False
    o.import_mesh = True
    o.static_mesh_import_data.combine_meshes = True
    o.static_mesh_import_data.auto_generate_collision = True          # False would DROP the UCX_ hulls (EagleKart trap)
    o.static_mesh_import_data.normal_import_method = unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS_AND_TANGENTS
    t.options = o
    at.import_asset_tasks([t])
    return unreal.load_asset(f"{dest}/{name}.{name}")


def make_mi(name, dest, master, texs, params, label, dflt):
    path = f"{dest}/{name}"
    mi = unreal.load_asset(f"{path}.{name}") if exists(path) else \
        at.create_asset(name, dest, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel = unreal.MaterialEditingLibrary
    mel.set_material_instance_parent(mi, master)
    for p, tx in texs.items():
        mel.set_material_instance_texture_parameter_value(mi, p, tx)
    mel.set_material_instance_texture_parameter_value(mi, "Label", label or dflt["clear"])
    mel.set_material_instance_scalar_parameter_value(mi, "UseLabel", 1.0 if label else 0.0)
    for k, v in params.items():
        if isinstance(v, (list, tuple)):
            mel.set_material_instance_vector_parameter_value(mi, k, unreal.LinearColor(r=v[0], g=v[1], b=v[2], a=1.0))
        else:
            mel.set_material_instance_scalar_parameter_value(mi, k, float(v))
    mel.update_material_instance(mi)
    eal.save_loaded_asset(mi)
    for k, v in params.items():                                      # read back, never trust the setter
        got = mel.get_material_instance_scalar_parameter_value(mi, k) if not isinstance(v, (list, tuple)) else None
        if got is not None and abs(got - float(v)) > 1e-4:
            unreal.log_warning(f"{name}.{k} = {got}, expected {v}")
    return mi


def import_prop(folder):
    man = json.load(open(os.path.join(folder, "manifest.json")))
    item = man["item"]
    dest = f"/Game/Lab/Props/{item}"
    dflt = defaults()                    # before the master: its samplers default to these
    master = lab_master.build()
    tdir = os.path.join(folder, "textures")
    role = {"BaseColor": "color", "Normal": "normal", "ORM": "masks", "Mask": "masks"}
    texs = {r: texture(f"{tdir}/T_{item}_{r}.png", f"{dest}/Textures", role[r]) for r in role}
    mis = {}
    for v, label_file in man["textures"]["variants"].items():
        lab = texture(os.path.join(tdir, label_file), f"{dest}/Textures", "label") if label_file else None
        mis[v] = make_mi(f"MI_{item}_{v}", f"{dest}/Materials", master, texs, man["variant_params"][v], lab, dflt)
    first = mis[next(iter(mis))]
    report = {}
    sems = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    fbxs = sorted(glob.glob(os.path.join(folder, "fbx", "*.fbx")))
    for f in [x for x in fbxs if "_LOD" not in x]:
        sm = import_mesh(f, dest)
        name = sm.get_name()
        for lod_f in sorted(x for x in fbxs if os.path.basename(x).startswith(name + "_LOD")):
            lod_i = int(lod_f.rsplit("_LOD", 1)[1][0])
            if sm.get_num_lods() <= lod_i:
                sems.import_lod(sm, lod_i, lod_f)
        for i in range(sm.get_num_sections(0)):
            sm.set_material(i, first)
        eal.save_loaded_asset(sm)
        report[name] = {"lods": sm.get_num_lods(), "tris": [sm.get_num_triangles(l) for l in range(sm.get_num_lods())],
                        "hulls": sm.get_editor_property("body_setup").get_editor_property("agg_geom").get_editor_property("convex_elems").__len__()
                        if sm.get_editor_property("body_setup") else 0,
                        "bounds_cm": [round(x, 2) for x in (sm.get_bounding_box().max - sm.get_bounding_box().min).to_tuple()]}
    report["materials"] = sorted(mis)
    json.dump(report, open(os.path.join(folder, "ue_import.json"), "w"), indent=1)
    unreal.log(f"imported {item}: {report}")


if __name__ == "__main__":
    for f in sys.argv[1:]:
        import_prop(os.path.abspath(f))
