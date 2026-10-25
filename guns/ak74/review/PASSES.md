# AK-74M - feature inventory and compare-and-fix passes

References (private, gitignored, never committed): `refs_user/ak74/` - right profile 163254 (master), left profile 163245,
high 3/4 mag-out 163300, folded 3/4 163304, close-ups 163309 (grip/stock R), 163315 (rear sight leaf), 163320 (dust cover,
RSB, serial, lower-handguard ribs, L), 163326 (rear sight from the left + hinge), 163334 (selector/trigger/grip, L side),
163337 (gas tube, handguards, barrel), 163339 (brake + FSB from the front), akt4anatomy.jpg.

## Calibration findings (before any modelling)
- Master scale: OAL 943 mm = columns 199.6..1179.4 -> 0.9624 mm/px.
- The master is rotated 0.554 deg (muzzle down): two barrel segments + the brake centres lie on one line (residual
  0.10 px). In that frame the dust-cover top is parallel to the bore (as on the rifle), so the B frame is rotated to match.
- The left photo is not an orthographic twin of the master: the muzzle end is ~5 % larger than the stock end (perspective,
  camera not square to the gun) and the whole photo is ~1.7 % wider. Each photo therefore gets a fitted camera
  (homography + radial term) in the side gate; the fit is reported (scale, rotation, perspective) next to every IoU.
- Mask repairs (a highlight is not a hole): specular streaks on the dust cover, the polished latch plate and bright pins
  are filled; the silver trigger guard is read at a lower coverage threshold; real see-through openings are kept (guard
  loop, gas-tube/barrel gap, rod gaps, FSB window, brake windows, leaf gap).

## Feature inventory per region (front to back)
1. **Muzzle brake**: collar r 12.75 (threaded section), chamber r 11.8, two large rectangular side windows through the front
   chamber, one small blind vent per side at the chamber rear, front lip r 8 with chamfered face, bottom slot in the lip
   (163339), wide inner chamber with a visible baffle ring, bore. Finish: parkerized, sand-textured (163339 shows grain).
2. **Front sight block**: barrel ring, tower leaning back, protective ears forming a U seen from the front with tips curling
   in, hooded post, round window through the tower, brake detent pin (bright dot), two taper pins, rod lug below
   (rod passes through), bayonet-style front lug.
3. **Gas block**: cup taking the gas tube (r 9.0), vertical web, barrel ring, two pin heads on the side (163337),
   lug below with the rod notch, arch over the barrel.
4. **Barrel**: r 6.4..6.7 visible, fine textured finish (163337), cleaning rod r 1.72 under it (slightly tilted in the
   photo), rod head in the FSB lug.
5. **Handguards + gas tube**: upper handguard = rounded-square polymer cover, top tapering forward, steel ferrules both
   ends; gas tube exposed front with a rear collar, 2 vent holes per side and shallow longitudinal flutes (163337); lower
   handguard rounded-rectangle with 6 longitudinal finger ribs at the rear (staggered fronts), 4 short vertical slots at the
   front, thumb bulge at the rear bottom, steel front retainer; gas-tube lock lever on the RSB right side.
6. **Receiver / dust cover**: stamped box, flat sides, tight bends; top-edge bead under the dust-cover skirt; ejection port
   (right) with the bright carrier visible; dust cover with flat top, big shoulder roll, a stamped step along the side,
   deeper front skirt (163320), guide-rod button through its rear; magazine-guide oval embossings both sides; front
   trunnion rivets (3 domes L, 3 R), rear trunnion small rivets, magwell rivet; bright trigger/hammer pins; generic serial
   band on the left lip (no factory proof mark: the triangle-arrow stamp in 163320 is deliberately NOT reproduced).
7. **Rear sight**: trapezoid block on the trunnion, leaf lying down with the rear notch tab, slider with knurled buttons
   both sides, range numbers 1-10 and the battle-setting letter (print mask), leaf/block see-through gap (x 725..757).
8. **Selector** (right): hub + axis nut, long tapered lever up-forward, bent finger tab covering the charging slot; three
   stop dimples above the hub (163300).
9. **Trigger group**: curved trigger blade (tip forward), stamped 1.8 mm guard (rear leg, flat bottom, front leg into the
   magazine-catch block), magazine catch paddle with its pin, the lever tip below.
10. **Pistol grip**: vertical front face at the top by the guard, straight front, palm swell, flared toe, rear tang under the
    receiver, checkered panel each side (material), bottom with the screw slot (163304).
11. **Magazine**: 30-rd polymer, concave front / convex rear arcs, slanted floorplate with lip, side panel step, small
    transverse tick ribs along spine and front (163300), front lug and rear catch tab on top. (163300's brass fill valve is
    an airsoft part - not modelled, see open items.)
12. **Stock + hinge**: polymer stock with the comb dip, right-side long recessed groove (163254), left-side framed panel and
    sling swivel (163245), right-front sling loop (163254/163309), buttplate with upper horn and lower lip, latch knob at
    the butt rear, left-side folding latch hook on the butt (163245/163304), hinge block with the vertical barrel on the
    left side, polished latch plate with a hole on the right rear of the receiver (163309).

## Pass 0 - coarse blockout (h x 2.5), side gate only
IoU right 0.9744 / left 0.9593 (camera-fitted), right 0.9642 before the frame rotation was found.
Differences found and the fixes going into pass 1:
- whole front end ran 0.5-1 px low: photo rotation 0.554 deg -> rotated B frame; all part coordinates converted point-wise.
- brake: vents were through-holes (photo: blind) -> blind; window size/position from crossings; cap radius 8.0 + chamfer.
- FSB: hand-drawn tower too narrow at the base / wrong top -> fitted outline (rms 0.26 px); window r 5.9 px.
- gas block: profile was a guess -> fitted outline (cup, front corner, slope, ring, lug; rms 0.34 px).
- gas tube: radius/axis from rotated-frame crossings (axis 24.3, r 8.35; cup r 9.0 on axis 25.4).
- upper handguard left a see-through slit over the lower one -> bottom now overlaps the lower handguard.
- rear sight: leaf/slider/block re-measured row by row (gap x 725.8..756.9, tower front x 774.1).
- trigger: hand polyline wrong -> fitted two-arc outline from row crossings; guard path re-measured (bottom 356.5).
- grip: missing vertical front face at the top (x 528.3, rows 318..345) and rear tang -> added.
- stock/hinge: receiver rear face is x 455.2 above the hinge block and 458.5 below; stock front x 452 -> fixed; the gap
  rows 278..285 and 312..319 is now open as in the photo.
- magazine rear edge top started 3 px too far forward -> vertex at (606, 322).
