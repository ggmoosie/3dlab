"""camera_match.py - recreate a reference photo's camera in Blender, then render the model from it ("spot the difference" step 1).

    blender -b <file.blend> --python camera_match.py -- <match.json>

match.json:
{
  "reference": "C:/.../refs/g5_right_side.png",          # the photo; its pixel size becomes the render size
  "camera": "REFCAM_right",                               # camera object to create / update
  "points": [                                             # >= 6 well-spread landmarks visible in the photo (4 is the minimum)
     {"name": "muzzle_top",  "px": [957, 92],  "world": [0.0, -0.093, 0.032]},   # world position in Blender units, or
     {"name": "rear_sight",  "px": [127, 70],  "object": "LM_rear_sight"},        # an Empty/object placed on the landmark, or
     {"name": "mag_toe",     "px": [272, 690], "vertex": ["SM_G17_Magazine", 812]} # a vertex index (world space)
  ],
  "init": {"focal_mm": 50, "view_dir": [1, 0, 0]},        # rough guess: direction FROM the camera TO the model, and lens
  "fix_focal": false,                                     # true if the lens is known (then focal_mm is used as is)
  "focal_range": [20, 400],                               # bounds for the solved lens (mm); product shots are usually 50-200
  # a point with "check": true is NOT used to solve, only reported - solve on the rigid, trusted landmarks (e.g. the slide and
  # sights), then the check points' errors are real shape differences between the model and the photo
  "render": {"out": "C:/.../review/render_right.png", "engine": "CYCLES", "samples": 64, "studio": true, "transparent": true}
}

Pixel coordinates: origin top-left of the photo, x right, y down (what any image viewer / PIL reports).
Solves camera rotation, position and focal length by Levenberg-Marquardt on the reprojection error, prints the per-point
error, creates the camera, sets the render size to the photo's and renders with a transparent background so
ref_compare.py can lay it over the photo. RMS error > 1 % of the image width means a landmark is wrong: fix it, rerun.

Gate (enforced): prints "CAM VERDICT PASS" or "CAM VERDICT FAIL: <reasons>". FAIL = rms over "max_rms_pct" (default 1.0) of the
width, or the solved focal stuck on a focal_range bound (when the lens is free). On FAIL the render is skipped and Blender
exits 2 (run Blender with --python-exit-code 1 so a crash is non-zero too). Optional keys: "max_rms_pct", "result" (path of a
JSON with the verdict, rms and per-point errors), "allow_unmatched": true (render anyway; landmark debugging only).
"""
import bpy, sys, json, math
import numpy as np
from mathutils import Vector, Matrix

cfg = json.load(open(sys.argv[sys.argv.index('--') + 1]))
img = bpy.data.images.load(cfg['reference'], check_existing=True); W, H = img.size
SENSOR = 36.0

def landmark(p):
    if 'world' in p: return np.array(p['world'], float)
    if 'object' in p: return np.array(bpy.data.objects[p['object']].matrix_world.translation[:], float)
    o = bpy.data.objects[p['vertex'][0]]; return np.array((o.matrix_world @ o.data.vertices[p['vertex'][1]].co)[:], float)

ALLP = np.array([landmark(p) for p in cfg['points']]); ALLUV = np.array([p['px'] for p in cfg['points']], float)
solve = np.array([not p.get('check', False) for p in cfg['points']]); P, UV = ALLP[solve], ALLUV[solve]
if len(P) < 4: sys.exit('camera_match: need at least 4 landmarks (6+ recommended)')

def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12: return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K

def project(x):
    R = rodrigues(x[:3]); t = x[3:6]; f = x[6] / SENSOR * W            # sensor_fit HORIZONTAL: focal in px from the width
    pc = (P - t) @ R                                                    # world -> camera (columns of R = camera axes)
    z = -pc[:, 2]; z = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return np.stack([W / 2 + f * pc[:, 0] / z, H / 2 - f * pc[:, 1] / z], 1), z

def residual(x):
    uv, z = project(x); r = (uv - UV).ravel()
    return np.concatenate([r, 1e3 * np.minimum(z, 0)])                 # points behind the camera are penalised

def look_rot(direction):
    d = np.array(direction, float); d /= np.linalg.norm(d)
    up = np.array([0, 0, 1.0]) if abs(d[2]) < 0.95 else np.array([0, 1.0, 0])
    zc = -d; xc = np.cross(up, zc); xc /= np.linalg.norm(xc); yc = np.cross(zc, xc)
    R = np.stack([xc, yc, zc], 1)
    ang = math.acos(max(-1, min(1, (np.trace(R) - 1) / 2)))
    if ang < 1e-9: return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.linalg.norm(v) < 1e-9:                                        # 180 deg: axis from the symmetric part
        w, V = np.linalg.eigh(R); v = V[:, np.argmax(w)]; return v / np.linalg.norm(v) * ang
    return v / np.linalg.norm(v) * ang

init = cfg.get('init', {}); fmm = float(init.get('focal_mm', 50.0)); vd = init.get('view_dir', [1, 0, 0])
c = P.mean(0); ext = np.linalg.norm(P.max(0) - P.min(0))
# start far enough that the landmarks fill about 60 % of the frame at the guessed lens
dist = ext / (0.6 * SENSOR / fmm)
x = np.concatenate([look_rot(vd), c - np.array(vd, float) / np.linalg.norm(vd) * dist, [fmm]])
FMIN, FMAX = cfg.get('focal_range', [20.0, 400.0])
free = np.ones(7, bool); free[6] = not cfg.get('fix_focal', False)
lam = 1e-2
for it in range(300):
    r = residual(x); J = np.zeros((len(r), 7))
    for k in np.where(free)[0]:
        h = 1e-6 * max(1.0, abs(x[k])); xp = x.copy(); xp[k] += h; J[:, k] = (residual(xp) - r) / h
    A = J.T @ J; g = J.T @ r
    step = np.zeros(7); idx = np.where(free)[0]
    step[idx] = -np.linalg.solve(A[np.ix_(idx, idx)] + lam * np.diag(np.diag(A)[idx] + 1e-9), g[idx])
    xn = x + step; xn[6] = min(max(FMIN, xn[6]), FMAX)
    if (residual(xn) ** 2).sum() < (r ** 2).sum(): x, lam = xn, lam * 0.3
    else: lam *= 10
    if np.linalg.norm(step) < 1e-10 or lam > 1e12: break

uv, z = project(x); err = np.linalg.norm(uv - UV, axis=1); rms = float(np.sqrt((err ** 2).mean()))
P, UV = ALLP, ALLUV; uva, _ = project(x)
for p, a, b in zip(cfg['points'], uva, ALLUV):
    print('CAM point %-16s %s error %6.1f px  (render %4.0f,%4.0f photo %4.0f,%4.0f)' % (p['name'], 'CHECK' if p.get('check') else 'solve', float(np.linalg.norm(a - b)), a[0], a[1], b[0], b[1]))
print('CAM rms %.2f px (%.2f %% of width) over the solve points, focal %.1f mm, iterations %d' % (rms, 100 * rms / W, x[6], it + 1))
# the gate is enforced, not just reported: a FAIL skips the render (an unmatched camera makes meaningless diff sheets) and
# exits 2. "max_rms_pct" (default 1.0) sets the limit; "allow_unmatched": true renders anyway, for debugging landmarks only.
fail = []
if rms > cfg.get('max_rms_pct', 1.0) / 100.0 * W:
    fail.append('rms %.2f %% of the width is over %.2f %% - a landmark pixel or 3D position is wrong, the model proportions differ (move suspect points to check), or the photo is distorted'
                % (100 * rms / W, cfg.get('max_rms_pct', 1.0)))
if not cfg.get('fix_focal', False) and (x[6] <= FMIN + 1e-3 or x[6] >= FMAX - 1e-3):
    fail.append('focal hit the focal_range bound - landmarks disagree; check each LM sits on the same feature as its pixel')
print('CAM VERDICT %s%s' % ('FAIL: ' if fail else 'PASS', '; '.join(fail)))
if cfg.get('result'):
    json.dump({'verdict': 'FAIL' if fail else 'PASS', 'reasons': fail, 'rms_px': rms, 'rms_pct': 100 * rms / W, 'focal_mm': float(x[6]), 'x0': [float(v) for v in x],
               'points': [{'name': p['name'], 'check': bool(p.get('check')), 'error_px': float(np.linalg.norm(a - b))} for p, a, b in zip(cfg['points'], uva, ALLUV)]},
              open(cfg['result'], 'w'), indent=1)

R = rodrigues(x[:3]); name = cfg.get('camera', 'REFCAM')
cam = bpy.data.objects.get(name)
if cam is None:
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name)); bpy.context.scene.collection.objects.link(cam)
M = Matrix([list(R[i]) + [x[3 + i]] for i in range(3)] + [[0, 0, 0, 1]]); cam.matrix_world = M
cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = SENSOR; cam.data.lens = float(x[6])
cam.data.clip_start = max(1e-4, float(z.min()) * 0.05); cam.data.clip_end = float(z.max()) * 20
sc = bpy.context.scene; sc.camera = cam; sc.render.resolution_x, sc.render.resolution_y = W, H; sc.render.resolution_percentage = 100

rc = cfg.get('render')
if rc and fail and not cfg.get('allow_unmatched', False):
    print('CAM render skipped: the camera is not matched (set "allow_unmatched": true to render anyway while debugging)')
elif rc:
    sc.render.engine = rc.get('engine', 'CYCLES')
    if sc.render.engine == 'CYCLES': sc.cycles.samples = int(rc.get('samples', 64)); sc.cycles.use_denoising = True
    sc.render.film_transparent = rc.get('transparent', True)
    if rc.get('studio'):                                                  # neutral 3-point light rig aimed at the landmarks
        for lname, off, power in (('QC_Key', (1.0, -1.2, 1.4), 1.0), ('QC_Fill', (-1.4, -0.6, 0.6), 0.45), ('QC_Rim', (0.0, 1.5, 1.0), 0.7)):
            lo = bpy.data.objects.get(lname)
            if lo is None:
                lo = bpy.data.objects.new(lname, bpy.data.lights.new(lname, 'AREA')); sc.collection.objects.link(lo)
            d = Vector(off).normalized() * ext * 3; lo.location = Vector(c) + d
            lo.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); lo.data.size = ext * 1.5
            lo.data.energy = power * 400 * (ext * 3) ** 2                   # roughly constant irradiance whatever the model size
    sc.render.image_settings.file_format = 'PNG'; sc.render.image_settings.color_mode = 'RGBA'
    sc.render.filepath = rc['out']; bpy.ops.render.render(write_still=True)
    print('CAM rendered', rc['out'])
if cfg.get('save_blend'): bpy.ops.wm.save_as_mainfile(filepath=cfg['save_blend'], copy=True)
if fail: sys.exit(2)
