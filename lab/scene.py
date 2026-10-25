"""Scene, studio and render helpers (bpy). Every prop script starts with reset() and ends with studio renders."""
import bpy, math, os
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.unit_settings.system = "METRIC"
    s.unit_settings.scale_length = 1.0
    return s


def collection(name, parent=None):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    p = parent or bpy.context.scene.collection
    if c.name not in [x.name for x in p.children]:
        p.children.link(c)
    return c


def link(obj, col):
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    col.objects.link(obj)
    return obj


def bounds(objs):
    pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def _area(name, loc, target, energy, size, color=(1, 1, 1), size_y=None):
    d = bpy.data.lights.new(name, "AREA")
    d.energy, d.color = energy, color
    d.shape = "RECTANGLE"
    d.size, d.size_y = size, size_y or size
    o = bpy.data.objects.new(name, d)
    bpy.context.scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return o


def studio(subject_objs, backdrop=True, scale=None, key=1.0):
    """Product-shot studio: soft key, fill, rim strip, grey sweep. Scales with the subject's size."""
    lo, hi = bounds(subject_objs)
    c, r = (lo + hi) / 2, max((hi - lo).length / 2, 0.01)
    k = scale or r / 0.08                      # rig was tuned around a 16 cm object
    _area("Key", c + Vector((0.35, -0.30, 0.42)) * k, c, 12 * k * k * key, 0.45 * k)
    _area("Fill", c + Vector((-0.45, -0.20, 0.15)) * k, c, 3.5 * k * k, 0.6 * k, (0.92, 0.96, 1.0))
    _area("Rim", c + Vector((-0.10, 0.45, 0.30)) * k, c, 9 * k * k, 0.12 * k, size_y=0.6 * k)
    w = bpy.data.worlds.new("Studio")
    bpy.context.scene.world = w
    w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.42, 0.43, 0.45, 1)
    w.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.25
    if backdrop:                                   # seamless cyclorama sweep: floor → radius → back wall
        import bmesh
        bm = bmesh.new()
        prof, R = [], 3 * r
        for k in range(9):
            prof.append((-8 * r + k * (8 * r - 0.0) / 8, 0.0))
        for k in range(1, 13):
            a_ = (math.pi / 2) * k / 12
            prof.append((R * math.sin(a_), R * (1 - math.cos(a_))))
        prof.append((R, 10 * r))
        rows = []
        for x in (-14 * r, 14 * r):
            rows.append([bm.verts.new((c.x + x, c.y + yy + 2 * r, lo.z + zz)) for yy, zz in prof])
        for k in range(len(prof) - 1):
            bm.faces.new((rows[0][k], rows[1][k], rows[1][k + 1], rows[0][k + 1]))
        me = bpy.data.meshes.new("Backdrop")
        bm.to_mesh(me)
        bm.free()
        p = bpy.data.objects.new("Backdrop", me)
        bpy.context.scene.collection.objects.link(p)
        for f in me.polygons:
            f.use_smooth = True
        m = bpy.data.materials.new("M_Backdrop")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (0.20, 0.205, 0.215, 1)
        b.inputs["Roughness"].default_value = 0.55
        p.data.materials.append(m)
    return c, r


def camera(target, dist, yaw_deg=-35, pitch_deg=18, lens=85, name="Cam"):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_start = 0.001
    o = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(o)
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    d = Vector((math.sin(y) * math.cos(p), -math.cos(y) * math.cos(p), math.sin(p)))
    o.location = Vector(target) + d * dist
    o.rotation_euler = (Vector(target) - o.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = o
    return o


def frame(objs, yaw=-35, pitch=18, lens=85, fill=0.72, name="Cam", aspect=1.0):
    """Camera that fits objs at the given angle (bounding sphere fit)."""
    lo, hi = bounds(objs)
    c, r = (lo + hi) / 2, (hi - lo).length / 2
    fov = 2 * math.atan(18 / lens) * min(1.0, aspect)   # 36 mm sensor across the wider side
    return camera(c, r / math.sin(fov / 2 * fill), yaw, pitch, lens, name)


def render(path, res=1024, samples=64, transparent=False, res_y=None):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = samples
    s.cycles.use_denoising = True
    s.render.resolution_x, s.render.resolution_y = res, res_y or res
    s.render.film_transparent = transparent
    s.view_settings.view_transform = "AgX"
    s.view_settings.look = "AgX - Punchy"
    s.view_settings.exposure = -0.35
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    s.render.filepath = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.render.render(write_still=True)
    return path


def tris(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    m = obj.evaluated_get(dg).to_mesh()
    n = sum(len(p.vertices) - 2 for p in m.polygons)
    obj.evaluated_get(dg).to_mesh_clear()
    return n
