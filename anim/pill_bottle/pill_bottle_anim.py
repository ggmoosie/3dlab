"""A_Consume_PillBottle: a NEW consumable clip (no existing clip covers it) on the UE5 Mannequin (Manny) skeleton.
  .venv/bin/python anim/pill_bottle/pill_bottle_anim.py [samples=24]
Action (4.0 s, 30 fps, 120 frames): hold -> raise bottle -> right thumb flips the cap -> tip two pills into the right palm ->
palm to mouth -> head back, swallow -> back to the hold. Bottle rides hand_l the whole clip (left hand carries the prop).
Rig: rebuilt from props/brief/skeleton_ue5_manny_rest.json (37 core bones, no fingers, bone roll 0). The game rig
(Tools/Anim/template_fp.blend, 1SK_Mannequin) has Epic's bone orientations, so this FBX is a REVIEW clip: retarget it onto
the template by bone name (world-space match) before import. The world-space wrist / head targets are also written to
<clip>.targets.json so the game side can bake them onto its own rig without guessing.
Out: out/A_Consume_PillBottle.fbx (armature only, contract FBX settings), .events.json, .targets.json,
     out/SM_PillBottle(+_Cap).fbx prop, renders/keyframes.jpg"""
import bpy, os, sys, json, math
import numpy as np
from mathutils import Vector, Matrix, Euler
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../..")
sys.path.insert(0, ROOT)
from lab import scene, geo, master

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
SMP = int(ARGS.get("samples", 24))
SKEL = "/home/user/3d/props/brief/skeleton_ue5_manny_rest.json"
OUT, REND = os.path.join(HERE, "out"), os.path.join(HERE, "renders")
FPS, N = 30, 120
CHAIN = {"pelvis": "spine_01", "spine_01": "spine_02", "spine_02": "spine_03", "spine_03": "spine_04", "spine_04": "spine_05",
         "spine_05": "neck_01", "neck_01": "neck_02", "neck_02": "head", "clavicle_l": "upperarm_l", "upperarm_l": "lowerarm_l",
         "lowerarm_l": "hand_l", "clavicle_r": "upperarm_r", "upperarm_r": "lowerarm_r", "lowerarm_r": "hand_r",
         "thigh_l": "calf_l", "calf_l": "foot_l", "foot_l": "ball_l", "thigh_r": "calf_r", "calf_r": "foot_r", "foot_r": "ball_r"}
UPPER = ["spine_02", "spine_03", "spine_04", "spine_05", "neck_01", "neck_02", "head", "clavicle_l", "upperarm_l", "lowerarm_l",
         "hand_l", "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r"]


def ue(v):
    """UE cm (x, y, z) -> Blender (x, -y, z) in cm units (scene scale 0.01)"""
    return Vector((v[0], -v[1], v[2]))


# ------------------------------------------------------------------ keys: UE cm, frame -> wrist targets + hand / head rotations
# hand rotation = extra rotation (deg, world XYZ in UE axes) on top of a neutral "palm-in, fingers forward" hand frame
KEYS = {
    #      frame: (hand_l pos, hand_l rot, hand_r pos, hand_r rot, head pitch, spine pitch)
    0:   ((9, 30, 108), (0, 0, 0), (-12, 26, 104), (0, 0, 0), 0, 0),
    14:  ((6, 33, 124), (0, 0, 0), (-7, 33, 125), (0, 0, -20), -12, 4),
    20:  ((6, 33, 125), (0, 0, 0), (-4, 34, 131), (0, -30, -35), -14, 4),     # thumb flicks the cap
    26:  ((6, 33, 125), (0, 0, 0), (-6, 32, 121), (0, 70, 0), -16, 5),        # palm turns up under the bottle
    38:  ((1, 35, 129), (0, -115, 0), (-6, 33, 120), (0, 75, 0), -18, 6),     # tip, pills fall into the palm
    46:  ((2, 35, 128), (0, -125, 0), (-6, 33, 120), (0, 75, 0), -18, 6),
    54:  ((9, 31, 116), (0, -10, 0), (-6, 33, 121), (0, 70, 0), -10, 4),
    64:  ((10, 30, 112), (0, 0, 0), (-1, 17, 155), (-40, 80, 10), 8, 0),       # palm to the mouth
    72:  ((10, 30, 112), (0, 0, 0), (-16, 27, 136), (0, 30, 0), 26, -4),      # head back, swallow
    86:  ((10, 30, 111), (0, 0, 0), (-14, 27, 120), (0, 10, 0), 20, -3),
    98:  ((9, 30, 109), (0, 0, 0), (-12, 26, 106), (0, 0, 0), 2, 0),
    119: ((9, 30, 108), (0, 0, 0), (-12, 26, 104), (0, 0, 0), 0, 0),
}
EVENTS = [(20, "UseItemStart"), (88, "UseItemEnd")]
CAP_KEYS = {0: 0.0, 19: 0.0, 22: 150.0, 119: 150.0}     # flip cap angle (deg) about its hinge


def build_rig():
    D = json.load(open(SKEL))["bones"]
    arm = bpy.data.armatures.new("Manny"); rig = bpy.data.objects.new("Manny", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for n, b in D.items():
        e = arm.edit_bones.new(n); e.head = ue(b["head"]); eb[n] = e
        if n in CHAIN:                                      # json tails are axis markers: chain bones run joint to joint
            e.tail = ue(D[CHAIN[n]]["head"])
        else:
            d = (ue(b["tail"]) - e.head).normalized()
            e.tail = e.head + d * 6.0
        e.roll = 0.0
    for side in "lr":                                       # hand: continue the forearm 9 cm; head: straight up
        h = eb["hand_" + side]; h.tail = h.head + (h.head - eb["lowerarm_" + side].head).normalized() * 9.0
    eb["head"].tail = eb["head"].head + Vector((0, 0, 16))
    for n, b in D.items():
        if b["parent"]:
            eb[n].parent = eb[b["parent"]]
            eb[n].use_connect = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig, D


def empty(name, loc):
    e = bpy.data.objects.new(name, None); e.empty_display_size = 4; e.location = loc
    bpy.context.scene.collection.objects.link(e)
    return e


def lerp_keys(keys, f, idx):
    fs = sorted(keys)
    for a, b in zip(fs, fs[1:]):
        if a <= f <= b:
            t = (f - a) / (b - a); t = t * t * (3 - 2 * t)
            va, vb = np.array(keys[a][idx], float), np.array(keys[b][idx], float)
            return va + (vb - va) * t
    return np.array(keys[fs[-1]][idx], float)


def animate(rig):
    s = bpy.context.scene
    s.render.fps = FPS; s.frame_start, s.frame_end = 0, N - 1
    pb = rig.pose.bones
    fwd = ue((0, 1, 0))                                     # neutral hand: fingers forward, bone Z up
    rest_l = rest_r = fwd.to_track_quat("Y", "Z")
    tl, tr = empty("T_hand_l", (0, 0, 0)), empty("T_hand_r", (0, 0, 0))
    pl, pr = empty("P_elbow_l", ue((45, 5, 90))), empty("P_elbow_r", ue((-45, 5, 90)))
    for side, t, p in (("l", tl, pl), ("r", tr, pr)):
        c = pb["lowerarm_" + side].constraints.new("IK"); c.target, c.pole_target, c.chain_count = t, p, 2
        c.pole_angle = math.radians(float(ARGS.get("pole", 90)))
        cr = pb["hand_" + side].constraints.new("COPY_ROTATION"); cr.target = t
        # the IK target is the hand bone's HEAD (wrist); put the target where the wrist should be
    for f in range(N):
        s.frame_set(f)
        for side, t, rest, ip, ir in (("l", tl, rest_l, 0, 1), ("r", tr, rest_r, 2, 3)):
            t.location = ue(lerp_keys(KEYS, f, ip))
            e = lerp_keys(KEYS, f, ir)
            rot = Euler((math.radians(e[0]), -math.radians(e[1]), math.radians(e[2])), "XYZ").to_quaternion()
            t.rotation_mode = "QUATERNION"; t.rotation_quaternion = rot @ rest
            t.keyframe_insert("location", frame=f); t.keyframe_insert("rotation_quaternion", frame=f)
        hp = lerp_keys(KEYS, f, 4); sp = lerp_keys(KEYS, f, 5)
        for n, w in (("neck_01", 0.35), ("neck_02", 0.25), ("head", 0.4)):
            pb[n].rotation_mode = "XYZ"; pb[n].rotation_euler = (math.radians(-hp * w), 0, 0); pb[n].keyframe_insert("rotation_euler", frame=f)
        for n, w in (("spine_03", 0.5), ("spine_04", 0.5)):
            pb[n].rotation_mode = "XYZ"; pb[n].rotation_euler = (math.radians(sp * w), 0, 0); pb[n].keyframe_insert("rotation_euler", frame=f)
    # bake to plain FK keys on the upper body, then drop the constraints and helpers
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")
    for b in rig.pose.bones:                                # Blender 5: selection lives on the pose bone
        b.select = b.name in UPPER
    bpy.ops.nla.bake(frame_start=0, frame_end=N - 1, only_selected=True, visual_keying=True, clear_constraints=True, bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.animation_data.action.name = "A_Consume_PillBottle"
    targets = {}
    for f in range(N):
        s.frame_set(f)
        targets[f] = {n: [round(c, 3) for c in ((rig.matrix_world @ pb[n].head) * Vector((1, -1, 1)))] for n in ("hand_l", "hand_r", "head")}
    for o in (tl, tr, pl, pr):
        bpy.data.objects.remove(o)
    return targets


def bottle():
    """Prop: 3.4 x 7.2 cm amber pill bottle with a flip cap (hinge at +Y rim). Pivot = base centre, +Z up (consume contract style)."""
    MM = 0.1                                                # scene units are cm: 1 mm = 0.1
    body = geo.lathe("SM_PillBottle", [dict(pts=[(0, 0), (16 * MM, 0), (17 * MM, 1 * MM), (17 * MM, 60 * MM), (15.5 * MM, 62 * MM),
                                                 (15.5 * MM, 66 * MM), (0, 66 * MM)], uv=(0, 0, 1, 1))], segs=32)
    cap = geo.lathe("SM_PillBottle_Cap", [dict(pts=[(0, 64 * MM), (18 * MM, 64 * MM), (18.5 * MM, 72 * MM), (17 * MM, 74 * MM), (0, 74 * MM)],
                                                uv=(0, 0, 1, 1))], segs=32)
    geo.set_origin(cap, (0, -17 * MM, 64 * MM))
    pills = []
    for i, (x, y) in enumerate(((0, 0), (0.6, 0.4))):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.45, location=(0, 0, 0), segments=16, ring_count=8)
        p = bpy.context.object; p.scale = (1, 1, 0.5); p.name = f"Pill{i}"; pills.append(p)
    amber = master.glass("M_PillBottle", tint=(0.85, 0.42, 0.08), rough=0.08)
    white = master.simple("M_PillCap", (0.85, 0.85, 0.83), 0.45, 0.0, None)
    pill = master.simple("M_Pill", (0.9, 0.9, 0.86), 0.5, 0.0, None)
    body.data.materials.append(amber); cap.data.materials.append(white)
    for p in pills:
        p.data.materials.append(pill)
    return body, cap, pills


def proxies(rig, D):
    """Capsule body proxy per bone, parented to the bone (review renders only)."""
    out = []
    R = {"pelvis": 12, "spine_01": 12, "spine_02": 13, "spine_03": 14, "spine_04": 15, "spine_05": 13, "neck_01": 5, "neck_02": 5,
         "clavicle_l": 5, "clavicle_r": 5, "upperarm_l": 5, "upperarm_r": 5, "lowerarm_l": 4, "lowerarm_r": 4, "hand_l": 3.4, "hand_r": 3.4,
         "thigh_l": 8, "thigh_r": 8, "calf_l": 6, "calf_r": 6, "foot_l": 4, "foot_r": 4}
    skin = master.simple("M_Proxy", (0.55, 0.55, 0.58), 0.5, 0.0, None)
    for n, r in R.items():
        b = rig.data.bones[n]
        h, t = rig.matrix_world @ b.head_local, rig.matrix_world @ b.tail_local
        if n.startswith(("upperarm", "lowerarm")):                       # arm bones end at the next joint
            child = {"upperarm_l": "lowerarm_l", "lowerarm_l": "hand_l", "upperarm_r": "lowerarm_r", "lowerarm_r": "hand_r"}[n]
            t = rig.matrix_world @ rig.data.bones[child].head_local
        if n.startswith("hand"):
            t = h + (t - h).normalized() * 9
        if n.startswith("thigh") or n.startswith("calf"):
            child = {"thigh_l": "calf_l", "calf_l": "foot_l", "thigh_r": "calf_r", "calf_r": "foot_r"}[n]
            t = rig.matrix_world @ rig.data.bones[child].head_local
        L = (t - h).length
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=max(L, 1), location=(h + t) / 2, vertices=16)
        c = bpy.context.object
        c.rotation_mode = "QUATERNION"; c.rotation_quaternion = (t - h).to_track_quat("Z", "Y")
        c.modifiers.new("b", "BEVEL").width = r * 0.9
        c.data.materials.append(skin)
        bpy.context.view_layer.update()
        mw = c.matrix_world.copy()
        c.parent, c.parent_type, c.parent_bone = rig, "BONE", n
        c.matrix_world = mw
        out.append(c)
    hb = rig.data.bones["head"]
    bpy.ops.mesh.primitive_uv_sphere_add(radius=10, location=rig.matrix_world @ hb.head_local + Vector((0, -2, 6)))
    hd = bpy.context.object; hd.scale = (0.85, 1.0, 1.1); hd.data.materials.append(skin)
    mw = hd.matrix_world.copy(); hd.parent, hd.parent_type, hd.parent_bone = rig, "BONE", "head"; hd.matrix_world = mw
    out.append(hd)
    return out


def main():
    scene.reset()
    s = bpy.context.scene
    s.unit_settings.system = "METRIC"; s.unit_settings.scale_length = 0.01      # 1 BU = 1 cm (contract: cm, no 0.01 wrapper)
    rig, D = build_rig()
    targets = animate(rig)
    if "probe" in ARGS:                                     # elbow height check for the pole angle
        for f in (0, 38, 64):
            bpy.context.scene.frame_set(f)
            pb = rig.pose.bones
            print("PROBE", ARGS.get("pole"), f, [round((rig.matrix_world @ pb[n].head).z - (rig.matrix_world @ pb[h].head).z, 1)
                                                  for n, h in (("lowerarm_l", "hand_l"), ("lowerarm_r", "hand_r"))])
        return
    body, cap, pills = bottle()
    for o in (body, cap):                                   # props exported in their own frame before parenting
        bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
        os.makedirs(OUT, exist_ok=True)
        bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, o.name + ".fbx"), use_selection=True, apply_scale_options="FBX_SCALE_NONE",
                                 axis_forward="-Y", axis_up="Z", object_types={"MESH"}, path_mode="STRIP", bake_anim=False,
                                 mesh_smooth_type="FACE", use_tspace=True, add_leaf_bones=False)
    # bottle in the left hand: hand frame at frame 0, bottle axis along the hand's palm-in grip
    s.frame_set(0)
    hl = rig.pose.bones["hand_l"]
    grip = Matrix.Translation((-2.0, 4.5, 0.5)) @ Euler((math.radians(90), 0, math.radians(90)), "XYZ").to_matrix().to_4x4()
    body.parent, body.parent_type, body.parent_bone = rig, "BONE", "hand_l"
    body.matrix_parent_inverse = Matrix.Identity(4)
    body.matrix_basis = Matrix.Translation((0, -rig.data.bones["hand_l"].length, 0)) @ grip
    cap.parent = body; cap.matrix_parent_inverse = Matrix.Identity(4)
    cap.rotation_mode = "XYZ"
    for f, a in CAP_KEYS.items():
        cap.rotation_euler = (math.radians(-a), 0, 0); cap.keyframe_insert("rotation_euler", frame=f)
    # pills: inside the bottle until the tip, then in the right palm until the mouth, then gone
    for i, p in enumerate(pills):
        p.parent, p.parent_type, p.parent_bone = rig, "BONE", "hand_r"
        p.matrix_parent_inverse = Matrix.Identity(4)
        p.location = (1.2 * i - 0.5, -rig.data.bones["hand_r"].length * 0.6, 2.4)
        for f, v in ((0, True), (39, True), (40, False), (66, False), (67, True)):
            p.hide_render = v; p.keyframe_insert("hide_render", frame=f)
    # export: armature only (contract settings), events, targets, prop
    os.makedirs(OUT, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT"); rig.select_set(True); bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, "A_Consume_PillBottle.fbx"), use_selection=True, apply_scale_options="FBX_SCALE_NONE",
                             axis_forward="-Y", axis_up="Z", armature_nodetype="NULL", add_leaf_bones=False, bake_anim=True,
                             bake_anim_use_all_actions=False, bake_anim_use_nla_strips=False, use_armature_deform_only=True,
                             object_types={"ARMATURE"}, path_mode="STRIP")
    json.dump({"fps": FPS, "events": [{"frame": f, "name": n} for f, n in EVENTS]}, open(os.path.join(OUT, "A_Consume_PillBottle.events.json"), "w"), indent=1)
    json.dump({"fps": FPS, "space": "UE component space, cm (character faces +Y, left = +X)", "frames": targets,
               "prop": {"mesh": "SM_PillBottle", "bone": "hand_l", "cap": "SM_PillBottle_Cap (hinge at the +Y rim, opens 150 deg on frames 19-22)",
                        "pills": "2 on hand_r frames 40-66"}}, open(os.path.join(OUT, "A_Consume_PillBottle.targets.json"), "w"), indent=1)
    # review renders: 6 key frames
    rig.data.pose_position = "REST"                         # proxies attach in the rest pose
    bpy.context.view_layer.update()
    prox = proxies(rig, D)
    rig.data.pose_position = "POSE"
    os.makedirs(REND, exist_ok=True)
    scene.studio(prox + [body])
    frames = (0, 20, 38, 64, 72, 119)
    paths = []
    for f in frames:
        s.frame_set(f)
        cam = scene.frame(prox, yaw=-30, pitch=8, lens=60, fill=0.9, name=f"c{f}", aspect=1.25)
        p = os.path.join(REND, f"f{f:03d}.png")
        scene.render(p, 640, SMP, res_y=800)
        paths.append(p)
    from PIL import Image, ImageDraw, ImageFont
    ims = [Image.open(p) for p in paths]
    sheet = Image.new("RGB", (640 * 3, 800 * 2 + 50), (22, 23, 26))
    d = ImageDraw.Draw(sheet)
    fb = ImageFont.truetype(os.path.join(ROOT, "fonts", "WorkSans-Bold.ttf"), 30)
    d.text((14, 8), "A_Consume_PillBottle - 4.0 s @ 30 fps (frames 0 / 20 cap / 38 tip / 64 mouth / 72 swallow / 119)", font=fb, fill=(235, 235, 235))
    for i, im in enumerate(ims):
        sheet.paste(im.convert("RGB"), ((i % 3) * 640, 50 + (i // 3) * 800))
    sheet.save(os.path.join(REND, "keyframes.jpg"), quality=90)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "pill_bottle_anim.blend"))
    print("ANIM_DONE")


if __name__ == "__main__":
    main()
