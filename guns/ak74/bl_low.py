"""bl_low.py - AK-74M game meshes from the SDF high-poly (bpy, Blender 5 headless; recipe = Glock v4 bl_low).
  per shell: copy -> quadric collapse to its budget -> weld / dissolve degenerate -> triangulate -> sliver cleanup
  per assembly: join -> weighted normals (face area) -> smart UVs packed in 0-1 (16 px margin at its texture size)
Highs stay in collection HIGH (bake source). usage: .venv/bin/python guns/ak74/bl_low.py -> work/ak_low.blend"""
import bpy, bmesh, sys, os, math, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "../.."))
import bl_high

ASM = {"Receiver": ["receiver", "rsb", "leaf", "slider", "hinge", "siderail", "guard", "magcatch", "rivets_rec"],
       "DustCover": ["dustcover"], "Recoil": ["recoil"], "Carrier": ["carrier", "bolt"],
       "Barrel": ["barrel", "gasblock", "fsb", "rivets_front"], "Muzzle": ["brake"], "GasTube": ["gastube", "upperhg", "hg_rings_u"],
       "LowerHG": ["lowerhg", "hg_ring_l"], "Rod": ["rod"], "Selector": ["selector"], "Trigger": ["trigger"], "Grip": ["grip"],
       "Mag": ["magazine"], "Stock": ["stock", "buttplate", "slings"]}
BUDGET = {"receiver": 5600, "rsb": 1400, "leaf": 500, "slider": 500, "hinge": 600, "siderail": 500, "guard": 600, "magcatch": 500,
          "rivets_rec": 0, "dustcover": 1900, "recoil": 900, "carrier": 1600, "bolt": 500, "barrel": 700, "gasblock": 1100,
          "fsb": 1500, "rivets_front": 0, "brake": 1700, "gastube": 900, "upperhg": 1100, "hg_rings_u": 600, "lowerhg": 2700,
          "hg_ring_l": 400, "rod": 250, "selector": 500, "trigger": 300, "grip": 1500, "magazine": 2600, "stock": 3200,
          "buttplate": 600, "slings": 500}
# 0 = bake-only (rivet and pin heads): they feed the normal / AO / ID bakes of their assembly, no game geometry
TEX = {"Receiver": 2048, "Barrel": 2048, "Stock": 2048, "LowerHG": 1024, "GasTube": 1024, "Mag": 1024, "Grip": 1024,
       "DustCover": 1024, "Muzzle": 1024, "Carrier": 1024, "Recoil": 512, "Rod": 512, "Selector": 512, "Trigger": 512}           # 2K maximum


def log(*a):
    print("[low]", *a, flush=True)


def tris(o):
    return sum(len(p.vertices) - 2 for p in o.data.polygons)


def coll(name):
    c = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if c.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(c)
    return c


def decimate_copy(src, target, col):
    me = src.data.copy()
    o = bpy.data.objects.new("LO_" + src.name[3:], me); col.objects.link(o)
    m = o.modifiers.new("dec", "DECIMATE"); m.decimate_type = "COLLAPSE"; m.ratio = min(1.0, target / max(1, tris(o)))
    m.use_collapse_triangulate = True
    with bpy.context.temp_override(object=o, active_object=o):
        bpy.ops.object.modifier_apply(modifier=m.name)
    bm = bmesh.new(); bm.from_mesh(o.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.dissolve_degenerate(bm, dist=2e-6, edges=bm.edges[:])
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(o.data); bm.free()
    o.data.polygons.foreach_set("use_smooth", [True] * len(o.data.polygons))
    return o


def _angles(f):
    vs = [v.co for v in f.verts]; out = []
    for i in range(3):
        u, w = vs[(i + 1) % 3] - vs[i], vs[(i + 2) % 3] - vs[i]
        out.append(u.angle(w) if u.length > 0 and w.length > 0 else 0.0)
    return out


def _is_sliver(f, deg=1.2, lmin=0.0018):
    if len(f.verts) != 3 or not f.is_valid:
        return False
    return min(_angles(f)) < math.radians(deg) and max(e.calc_length() for e in f.edges) > lmin


def fix_slivers(o, max_iter=12, max_collapse=0.0006):
    """Glock v4 recipe: flip caps, collapse needles (link condition), beautify flat edges; revert if non-manifold."""
    bm = bmesh.new(); bm.from_mesh(o.data); backup = o.data.copy()
    for it in range(max_iter):
        bm.faces.ensure_lookup_table(); flips, cols = set(), set()
        for f in bm.faces:
            if not _is_sliver(f):
                continue
            a = _angles(f)
            if max(a) > math.radians(150):
                e = max(f.edges, key=lambda e: e.calc_length())
                if len(e.link_faces) == 2:
                    flips.add(e)
            else:
                e = min(f.edges, key=lambda e: e.calc_length())
                lim = max_collapse * (1.7 if max(x.calc_length() for x in f.edges) > 0.012 else 1.0)
                if e.calc_length() > lim or len(e.link_faces) != 2:
                    continue
                v0, v1 = e.verts
                n0 = {x.other_vert(v0) for x in v0.link_edges}; n1 = {x.other_vert(v1) for x in v1.link_edges}
                if len(n0 & n1) != 2:
                    continue
                if any(x in cols for x in v0.link_edges) or any(x in cols for x in v1.link_edges):
                    continue
                cols.add(e)
        if not flips and not cols:
            break
        for e in flips:
            if not e.is_valid or len(e.link_faces) != 2:
                continue
            a = [v for v in e.link_faces[0].verts if v not in e.verts]; b = [v for v in e.link_faces[1].verts if v not in e.verts]
            if len(a) != 1 or len(b) != 1 or any(x.other_vert(a[0]) is b[0] for x in a[0].link_edges):
                continue
            try:
                bmesh.utils.edge_rotate(e, False)
            except Exception:
                pass
        cols = [e for e in cols if e.is_valid]
        if cols:
            bmesh.ops.collapse(bm, edges=cols, uvs=False)
        bmesh.ops.dissolve_degenerate(bm, dist=2e-6, edges=bm.edges[:])
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    flat = [e for e in bm.edges if len(e.link_faces) == 2 and e.calc_face_angle(0.0) < math.radians(2.0)]
    if flat:
        bmesh.ops.beautify_fill(bm, faces=bm.faces[:], edges=flat, use_restrict_tag=False, method="ANGLE")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    left = sum(1 for f in bm.faces if _is_sliver(f, 1.0, 0.002))
    bad = sum(1 for e in bm.edges if len(e.link_faces) != 2) + sum(1 for v in bm.verts if not v.is_manifold)
    if bad:
        bm.free(); old = o.data; o.data = backup; bpy.data.meshes.remove(old)
        log("  sliver cleanup reverted on", o.name, bad); return -1
    bm.to_mesh(o.data); bm.free(); bpy.data.meshes.remove(backup)
    return left


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    hcol, gcol = coll("HIGH"), coll("GAME")
    highs = bl_high.load()
    for o in highs.values():
        for c in list(o.users_collection):
            c.objects.unlink(o)
        hcol.objects.link(o)
    report = {}
    t0 = time.time()
    for asm, shells in ASM.items():
        objs = []
        for s in shells:
            if BUDGET[s] == 0:
                log(f"{s:10s} bake-only (no game geometry)")
                continue
            t = time.time()
            o = decimate_copy(highs[s], BUDGET[s], gcol)
            left = fix_slivers(o)
            log(f"{s:10s} {tris(highs[s]):8d} -> {tris(o):6d}  slivers {left}  {time.time() - t:.1f}s")
            objs.append(o)
        with bpy.context.temp_override(active_object=objs[0], selected_editable_objects=objs, object=objs[0]):
            bpy.ops.object.join()
        o = objs[0]
        name = "SM_AK74_" + asm
        o.name = o.data.name = name
        o.data.materials.clear()
        bm = bmesh.new(); bm.from_mesh(o.data); bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bmesh.ops.dissolve_degenerate(bm, dist=1e-5, edges=bm.edges[:])
        # game-mesh hygiene: close small holes left by decimation/clipping, drop tiny detached islands (< 60 faces)
        bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=16)
        seen, drop = set(), []
        for f0 in bm.faces:
            if f0.index in seen:
                continue
            st, isl = [f0], []
            seen.add(f0.index)
            while st:
                f = st.pop(); isl.append(f)
                for e in f.edges:
                    for g in e.link_faces:
                        if g.index not in seen:
                            seen.add(g.index); st.append(g)
            if len(isl) < 60:
                drop += isl
        if drop:
            bmesh.ops.delete(bm, geom=list(set(drop)), context="FACES")
            log(f"  {asm}: dropped {len(set(drop))} faces of tiny islands")
        bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
        bm.to_mesh(o.data); bm.free()
        w = o.modifiers.new("wn", "WEIGHTED_NORMAL"); w.mode = "FACE_AREA"; w.weight = 50; w.keep_sharp = False; w.thresh = 0.01
        with bpy.context.temp_override(object=o, active_object=o):
            bpy.ops.object.modifier_apply(modifier=w.name)
        size = TEX[asm]
        for x in bpy.context.view_layer.objects:
            x.select_set(False)
        o.select_set(True); bpy.context.view_layer.objects.active = o
        if not o.data.uv_layers:
            o.data.uv_layers.new(name="UVMap")
        bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(40 if asm in ("Barrel", "Muzzle") else 60), island_margin=0.0,
                                 area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(rotate=True, scale=True, margin_method="FRACTION", margin=16.0 / size, shape_method="CONCAVE")
        bpy.ops.object.mode_set(mode="OBJECT")
        m = bpy.data.materials.get("MI_AK74_" + asm) or bpy.data.materials.new("MI_AK74_" + asm)
        o.data.materials.append(m)
        o["tex_size"] = size
        o["shells"] = ",".join(shells)
        report[name] = dict(tris=tris(o), verts=len(o.data.vertices), tex=size)
        log("ASSEMBLY", name, report[name])
    hcol.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "work", "ak_low.blend"))
    json.dump(report, open(os.path.join(HERE, "work", "low_report.json"), "w"), indent=1)
    log(f"LOW_DONE {time.time() - t0:.0f}s LOD0 tris {sum(v['tris'] for v in report.values())}")


if __name__ == "__main__":
    main()
