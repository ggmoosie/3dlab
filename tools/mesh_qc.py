"""mesh_qc.py - Blender mesh quality gate (Blender 4.x, headless). Every 3D deliverable passes this before anyone looks at it.

    blender -b <file.blend> --python mesh_qc.py -- [options]

Options (all optional):
    --objects A,B        only these mesh objects (default: every visible mesh object)
    --collection NAME    only meshes in this collection (recursive)
    --open A,B           objects allowed to have open boundaries (decals, cards, one-sided floor plates); or set the custom
                         property qc_open=True on the object
    --allow-float A,B    objects whose loose shells are intended (e.g. an exploded view, particles)
    --uv-overlap-ok A,B  objects whose UV0 deliberately stacks islands (mirrored halves, trim sheets, tiling)
    --uv-tile            UV0 may leave 0-1 (tiling materials); default is baked-texture rules (0-1 only)
    --no-uv              skip UV checks (blockouts)
    --budget N           max triangles over all checked objects
    --tol-mm F           contact tolerance for floating-part / z-fight detection (default 0.1 mm)
    --per-object-float   floating parts are judged per object (a kit of separate props), not against one assembly
    --raw                check the raw mesh data instead of the evaluated (modifiers applied) mesh
    --json PATH          write every finding (with world positions in mm) to PATH

Exit code 1 when any ERROR is found. Output lines: "QC ERROR|WARN|INFO <object>: <finding>".
Units: findings are in mm, assuming 1 Blender unit = scene.unit_settings.scale_length metres.
"""
import bpy, bmesh, sys, json, math
import numpy as np
from mathutils import Vector, kdtree
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
def opt(name, default=None, flag=False):
    if name not in argv: return default
    if flag: return True
    i = argv.index(name) + 1
    if i >= len(argv) or argv[i].startswith('--'): sys.exit('mesh_qc: %s needs a value (see the options in the docstring)' % name)
    return argv[i]
def opt_set(name): v = opt(name); return set(v.split(',')) if v else set()

SCALE_MM = bpy.context.scene.unit_settings.scale_length * 1000.0      # Blender units -> mm
TOL_MM = float(opt('--tol-mm', 0.1)); TOL = TOL_MM / SCALE_MM
OPEN, FLOAT_OK, UVOV_OK = opt_set('--open'), opt_set('--allow-float'), opt_set('--uv-overlap-ok')
findings = []
def report(level, obj, msg, where=None):
    findings.append({'level': level, 'object': obj, 'msg': msg, 'where_mm': [[round(c * SCALE_MM, 2) for c in p] for p in (where or [])[:20]]})
    print('QC %s %s: %s' % (level, obj, msg))

# ---------------------------------------------------------------------------------------------------- object selection
def pick_objects():
    names = opt_set('--objects'); col = opt('--collection')
    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH' and o.visible_get()]
    if names: objs = [o for o in objs if o.name in names]
    if col:
        c = bpy.data.collections.get(col)
        if not c: sys.exit('mesh_qc: no collection %s' % col)
        inside = set(c.all_objects); objs = [o for o in objs if o in inside]
    return objs

def world_bmesh(o):
    """the mesh as exported: evaluated (modifiers applied) unless --raw, in world space"""
    bm = bmesh.new()
    if opt('--raw', flag=True): bm.from_mesh(o.data)
    else:
        dg = bpy.context.evaluated_depsgraph_get(); oe = o.evaluated_get(dg); me = oe.to_mesh(); bm.from_mesh(me); oe.to_mesh_clear()
    bm.transform(o.matrix_world); bm.normal_update()
    for seq in (bm.verts, bm.edges, bm.faces): seq.ensure_lookup_table()
    return bm

def shells_of(bm):
    comp = [-1] * len(bm.faces); shells = []
    for f0 in bm.faces:
        if comp[f0.index] >= 0: continue
        stack, members = [f0], []; comp[f0.index] = len(shells)
        while stack:
            f = stack.pop(); members.append(f.index)
            for e in f.edges:
                for g in e.link_faces:
                    if comp[g.index] < 0: comp[g.index] = len(shells); stack.append(g)
        shells.append(members)
    return shells, comp

# ---------------------------------------------------------------------------------------------------- per-object checks
def check_object(o, bm, shells, comp):
    n = o.name
    if any(s < 0 for s in o.scale): report('ERROR', n, 'negative scale %s (flips normals on export): apply scale' % (tuple(round(s, 3) for s in o.scale),))
    elif any(abs(s - 1) > 1e-4 for s in o.scale): report('WARN', n, 'unapplied scale %s' % (tuple(round(s, 3) for s in o.scale),))
    loose_v = [v for v in bm.verts if not v.link_edges]; loose_e = [e for e in bm.edges if not e.link_faces]
    if loose_v: report('ERROR', n, '%d loose vertices' % len(loose_v), [v.co for v in loose_v])
    if loose_e: report('ERROR', n, '%d loose edges (wire)' % len(loose_e), [(e.verts[0].co + e.verts[1].co) / 2 for e in loose_e])
    degen = [f for f in bm.faces if f.calc_area() < (0.001 / SCALE_MM) ** 2]
    if degen: report('ERROR', n, '%d zero-area faces (shading spikes, bake artefacts): dissolve degenerate' % len(degen), [f.calc_center_median() for f in degen])
    short = [e for e in bm.edges if e.calc_length() < 0.001 / SCALE_MM]
    if short: report('ERROR', n, '%d zero-length edges: merge by distance' % len(short), [e.verts[0].co for e in short])
    kd = kdtree.KDTree(len(bm.verts))
    for v in bm.verts: kd.insert(v.co, v.index)
    kd.balance()
    dup = [v for v in bm.verts if len(kd.find_range(v.co, 0.01 / SCALE_MM)) > 1]
    if dup: report('WARN', n, '%d coincident vertices (unwelded seams -> cracks / hard shading lines). Merge unless the split is a deliberate UV/hard-edge split' % len(dup), [v.co for v in dup])
    # topology
    multi = [e for e in bm.edges if len(e.link_faces) > 2]
    if multi: report('ERROR', n, '%d non-manifold edges (3+ faces share an edge)' % len(multi), [(e.verts[0].co + e.verts[1].co) / 2 for e in multi])
    bound = [e for e in bm.edges if len(e.link_faces) == 1]
    if bound:
        lvl = 'INFO' if (n in OPEN or o.get('qc_open')) else 'ERROR'
        report(lvl, n, '%d open boundary edges (holes; not airtight). Close them, or mark the object --open / qc_open if it is a card/decal' % len(bound),
               [(e.verts[0].co + e.verts[1].co) / 2 for e in bound])
    flipped = []
    for e in bm.edges:
        if len(e.link_faces) != 2: continue
        f1, f2 = e.link_faces; a, b = e.verts
        def dirn(f):
            vs = list(f.verts); i = vs.index(a); return vs[(i + 1) % len(vs)] is b
        if dirn(f1) == dirn(f2): flipped.append((a.co + b.co) / 2)
    if flipped: report('ERROR', n, '%d edges between faces of opposite winding (flipped faces): recalculate normals outside' % len(flipped), flipped)
    for si, faces in enumerate(shells):
        closed = all(len(e.link_faces) == 2 for fi in faces for e in bm.faces[fi].edges)
        if not closed: continue
        # signed volume about the shell's own centroid: summed about the world origin, a 2 cm part 20 m away loses its sign to
        # float error (2026-10-24, GunRange kit catalogue: 3 false "inside-out" shells)
        vs = {v.index: v.co for fi in faces for v in bm.faces[fi].verts}
        cen = sum(vs.values(), Vector()) / max(len(vs), 1)
        vol = 0.0
        for fi in faces:
            pts = [v.co - cen for v in bm.faces[fi].verts]
            for i in range(1, len(pts) - 1): vol += pts[0].dot(pts[i].cross(pts[i + 1]))
        if vol < 0: report('ERROR', n, 'shell %d is inside-out (negative volume): flip normals' % si, [bm.faces[faces[0]].calc_center_median()])
    # shading-risk geometry
    nonplanar, slivers, ngons = [], [], 0
    for f in bm.faces:
        k = len(f.verts)
        if k > 4: ngons += 1
        if k >= 4:
            c = f.calc_center_median(); nrm = f.normal; size = max((v.co - c).length for v in f.verts)
            dev = max(abs((v.co - c).dot(nrm)) for v in f.verts)
            if dev * SCALE_MM > 0.05 and dev > 0.01 * size: nonplanar.append(c)
        if k == 3:
            vs = [v.co for v in f.verts]; ang = []
            for i in range(3):
                u, w = vs[(i + 1) % 3] - vs[i], vs[(i + 2) % 3] - vs[i]
                if u.length > 0 and w.length > 0: ang.append(u.angle(w))
            if ang and min(ang) < math.radians(1.0) and max(u.length for u in (vs[1] - vs[0], vs[2] - vs[1], vs[0] - vs[2])) * SCALE_MM > 2.0:
                slivers.append(f.calc_center_median())
    if nonplanar: report('WARN', n, '%d non-planar quads/n-gons (diagonal shading seams): triangulate or add support edges' % len(nonplanar), nonplanar)
    if slivers: report('WARN', n, '%d long sliver triangles (<1 deg; boolean debris, the "spike" look): clean the boolean / rebuild the patch' % len(slivers), slivers)
    if ngons: report('INFO', n, '%d n-gons (fine on flat hidden faces; never on curved or silhouette surfaces)' % ngons)
    tris = sum(len(f.verts) - 2 for f in bm.faces)
    me = o.data
    if not opt('--raw', flag=True) and not me.has_custom_normals and not any(m.type == 'WEIGHTED_NORMAL' for m in o.modifiers):
        report('INFO', n, 'no custom / weighted normals (hard-surface: add a Weighted Normal modifier or custom normals)')
    slots = [s.material.name if s.material else None for s in o.material_slots]
    if not slots or None in slots: report('WARN', n, 'empty material slot(s) %s' % slots)
    if not opt('--no-uv', flag=True): check_uvs(o, bm, shells)
    return tris

def check_uvs(o, bm, shells):
    n = o.name; uvl = bm.loops.layers.uv.active
    if uvl is None: report('ERROR', n, 'no UV map'); return
    uv_tris_by_mat, zero_uv, flipped, dens = {}, [], [0, 0], []
    for f in bm.faces:
        uvs = [l[uvl].uv.copy() for l in f.loops]; a3 = f.calc_area()
        ua = 0.0
        for i in range(1, len(uvs) - 1):
            t = (uvs[0], uvs[i], uvs[i + 1]); uv_tris_by_mat.setdefault(f.material_index, []).append([tuple(p) for p in t])
            ua += ((t[1].x - t[0].x) * (t[2].y - t[0].y) - (t[2].x - t[0].x) * (t[1].y - t[0].y)) / 2
        if a3 > (0.5 / SCALE_MM) ** 2 and abs(ua) < 1e-10: zero_uv.append(f.calc_center_median())
        if abs(ua) > 1e-10:
            flipped[0 if ua > 0 else 1] += 1
            dens.append(math.sqrt(abs(ua) / a3) / SCALE_MM if a3 > 0 else 0)       # UV units per mm
    if zero_uv: report('ERROR', n, '%d faces with zero UV area (texture streaks): unwrap them' % len(zero_uv), zero_uv)
    if min(flipped) > 0: report('WARN', n, '%d of %d UV faces are mirrored (normal-map seams unless the mirror is deliberate)' % (min(flipped), sum(flipped)))
    if dens:
        d = np.array([x for x in dens if x > 0]); lo, hi = np.percentile(d, 5), np.percentile(d, 95)
        if lo > 0 and hi / lo > 2.5: report('WARN', n, 'texel density spread %.1fx between 5th and 95th percentile faces: rescale islands to a consistent density' % (hi / lo))
    allu = np.array([p for tris in uv_tris_by_mat.values() for t in tris for p in t])
    if not opt('--uv-tile', flag=True) and allu.size and ((allu < -1e-4).any() or (allu > 1 + 1e-4).any()):
        report('ERROR', n, 'UV0 outside 0-1 (baked textures would wrap): pack into 0-1, or pass --uv-tile for tiling materials')
    if n in UVOV_OK: return
    for mi, tris in uv_tris_by_mat.items():
        ov = overlap_pct(np.array(tris, dtype=np.float64) % 1.0 if opt('--uv-tile', flag=True) else np.array(tris, dtype=np.float64))
        if ov > 0.5:
            slot = o.material_slots[mi].material.name if mi < len(o.material_slots) and o.material_slots[mi].material else mi
            report('ERROR', n, 'UV overlap %.2f %% in material %s (bakes and wear masks double up): separate islands, or --uv-overlap-ok if stacking is deliberate' % (ov, slot))

def overlap_pct(uv_tris, res=1024):
    """rasterise UV triangles (shrunk 0.35 px toward their centroid so shared edges never double count) -> % of covered texels hit twice"""
    cnt = np.zeros((res, res), np.uint16); T = uv_tris * res
    for tri in T:
        c = tri.mean(0); r = max(1e-6, np.linalg.norm(tri - c, axis=1).max()); tri = c + (tri - c) * max(0.0, 1.0 - 0.35 / r)
        x0, y0 = np.floor(tri.min(0)).astype(int); x1, y1 = np.ceil(tri.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, res), min(y1, res)
        if x1 <= x0 or y1 <= y0: continue
        xs, ys = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        (ax, ay), (bx, by), (cx, cy) = tri
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12: continue
        l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / d; l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / d
        cnt[y0:y1, x0:x1] += ((l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)).astype(np.uint16)
    cov = (cnt >= 1).sum(); return 100.0 * (cnt >= 2).sum() / max(1, cov)

# ---------------------------------------------------------------------------------------------------- assembly checks
class Shell:
    def __init__(self, obj, idx, bm, faces):
        self.obj, self.idx = obj, idx
        vidx = sorted({v.index for fi in faces for v in bm.faces[fi].verts}); remap = {v: i for i, v in enumerate(vidx)}
        self.co = [bm.verts[v].co.copy() for v in vidx]
        self.polys = [[remap[v.index] for v in bm.faces[fi].verts] for fi in faces]
        self.faces = [(bm.faces[fi].calc_center_median(), bm.faces[fi].normal.copy(), bm.faces[fi].calc_area()) for fi in faces]
        self.bvh = BVHTree.FromPolygons(self.co, self.polys, epsilon=0.0)
        a = np.array([c[:] for c in self.co]); self.lo, self.hi = a.min(0), a.max(0)
        self.size = float(np.linalg.norm(self.hi - self.lo)); self.label = '%s#%d' % (obj, idx)
    def near(self, other, pad):
        return (self.lo - pad <= other.hi).all() and (other.lo - pad <= self.hi).all()

def assembly_checks(all_shells):
    """floating parts: every shell must touch or intersect the rest (graph connected to the biggest shell of its object group).
    z-fighting: a face whose centre lies on another shell's face (within tol) with the SAME facing = two visible coplanar surfaces."""
    N = len(all_shells); adj = [set() for _ in range(N)]; zf = {}
    for i in range(N):
        A = all_shells[i]
        for j in range(i + 1, N):
            B = all_shells[j]
            if not A.near(B, TOL): continue
            touching = bool(A.bvh.overlap(B.bvh))
            if not touching:
                small, big = (A, B) if len(A.co) <= len(B.co) else (B, A)
                step = max(1, len(small.co) // 600)
                touching = any(big.bvh.find_nearest(p, TOL)[0] is not None for p in small.co[::step])
                if not touching:
                    step = max(1, len(small.faces) // 600)
                    touching = any(big.bvh.find_nearest(c, TOL)[0] is not None for c, _, _ in small.faces[::step])
            if touching: adj[i].add(j); adj[j].add(i)
            for P, Q in ((A, B), (B, A)):                   # z-fight: P's faces lying on Q's surface, same facing
                for c, nrm, area in P.faces:
                    if area * SCALE_MM ** 2 < 1.0: continue           # ignore < 1 mm^2 faces
                    hit = Q.bvh.find_nearest(c, TOL)
                    if hit[0] is not None and hit[1].dot(nrm) > 0.999: zf.setdefault((P.label, Q.label), []).append(c)
    for (p, q), pts in zf.items():
        report('ERROR', p, '%d faces coplanar with %s and facing the same way (z-fighting): recess, raise >= 0.2 mm, or delete the hidden copy' % (len(pts), q), pts)
    # floating: per object group (objects in --allow-float skip), components not containing the largest shell
    # --per-object-float: each object is its own assembly (a kit of separate props); default = one assembly for everything checked
    per_obj = opt('--per-object-float', flag=True)
    seen = [False] * N; comps = []
    for s in range(N):
        if seen[s]: continue
        stack, members = [s], []; seen[s] = True
        while stack:
            k = stack.pop(); members.append(k)
            for m in adj[k]:
                if not seen[m] and (not per_obj or all_shells[m].obj == all_shells[k].obj): seen[m] = True; stack.append(m)
        comps.append(members)
    groups = {}
    for c in comps:
        groups.setdefault(all_shells[c[0]].obj if per_obj else '*', []).append(c)
    for gcomps in groups.values():
        if len(gcomps) < 2: continue
        main = max(gcomps, key=lambda c: sum(all_shells[k].size for k in c))
        for c in gcomps:
            if c is main: continue
            objs = {all_shells[k].obj for k in c}
            if objs <= FLOAT_OK: continue
            labels = [all_shells[k].label for k in c][:6]
            lo = np.min([all_shells[k].lo for k in c], 0); hi = np.max([all_shells[k].hi for k in c], 0)
            report('ERROR', ','.join(sorted(objs)), 'floating part(s) %s (size %.1f mm) touch nothing within %.2f mm: seat it on / into its support' %
                   (labels, float(np.linalg.norm(hi - lo)) * SCALE_MM, TOL_MM), [Vector(((lo + hi) / 2).tolist())])

# ---------------------------------------------------------------------------------------------------- main
objs = pick_objects()
if not objs: sys.exit('mesh_qc: no mesh objects to check')
total, all_shells = 0, []
for o in objs:
    bm = world_bmesh(o); shells, comp = shells_of(bm)
    total += check_object(o, bm, shells, comp)
    all_shells += [Shell(o.name, i, bm, f) for i, f in enumerate(shells)]
    bm.free()
assembly_checks(all_shells)
budget = opt('--budget')
if budget and total > int(budget): report('ERROR', '*', 'triangles %d over budget %s' % (total, budget))
ne = sum(f['level'] == 'ERROR' for f in findings); nw = sum(f['level'] == 'WARN' for f in findings)
print('QC SUMMARY objects=%d shells=%d tris=%d errors=%d warnings=%d -> %s' % (len(objs), len(all_shells), total, ne, nw, 'FAIL' if ne else 'PASS'))
if opt('--json'):
    json.dump({'objects': [o.name for o in objs], 'tris': total, 'errors': ne, 'warnings': nw, 'findings': findings}, open(opt('--json'), 'w'), indent=1)
sys.stdout.flush()
if ne: sys.exit(1)
