"""outlines.py - AK-74M side outlines (photo-R px, read off the pixel grids pxgrid.py at 4-8x) as clean parametric polygons:
straight edges + circular arcs (sagitta, px) + corner fillets (mm). Edges on the photo silhouette (sil=1) are snapped to the
photo edge by least squares (ak_ref.Outline.fit); internal edges (seams between parts, hidden ends) stay as measured.
Every vertex comment says where it was read."""
import numpy as np
import ak_ref as R

O = R.Outline


def _o(name, pts, r, sil, sag=0.0, fix=()):
    return name, O(pts, r, sil, sag, name, fix)


def table():
    T = dict([
        # ---------------- stock (polymer). top 284.8 -> 283.0 (x 206..346), comb dip to 287.8 at x 374, rises to 278 at
        # x 440; bottom: heel round x 202..226, butt line slope -0.35 to x 345, wrist line -0.215 to the front.
        # front face x 452 (rows 278..286, then the hinge block), lower front reaches x 455.5 (rows 312..322)
        _o("stock", [(199.6, 284.9), (346.0, 283.0), (374.0, 287.9), (440.0, 278.0), (450.2, 277.9), (452.0, 279.5),
                     (452.0, 310.0), (455.5, 311.5), (455.5, 321.5), (444.0, 322.6), (346.0, 343.2), (210.0, 389.0),
                     (199.6, 388.4)],
           [1.0, 10.0, 14.0, 6.0, 1.0, 0.6, 0.6, 0.5, 0.6, 1.0, 40.0, 7.0, 1.0],
           [1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0], fix=(0, 12, 4, 5, 6, 7, 8)),
        # buttplate: rear with the upper horn (x 194..199.6, rows 285..316) and a low lip (rows 368..388)
        _o("buttplate", [(194.3, 286.0), (199.8, 284.9), (199.8, 388.4), (198.0, 388.0), (198.0, 368.5), (199.0, 367.0),
                         (199.0, 317.5), (197.0, 316.0), (194.3, 293.5)],
           [0.8, 0, 0, 0.5, 0.4, 0, 0, 0.8, 0.8], [0, 0, 0, 1, 0, 0, 0, 1, 1], fix=(1, 2)),
        # ---------------- receiver body (stamped). Rear face x 455.2 at the top (rows 257..286, rear trunnion cap), x 458.5
        # below the hinge block (rows 300..319); bottom 315.6 between the guard ends, 310.2..312.5 at the magwell front;
        # front inside the RSB / lower handguard at x 727
        _o("receiver", [(455.2, 257.0), (700.0, 257.0), (727.0, 262.0), (727.0, 310.0), (704.0, 310.5), (700.0, 312.5),
                        (680.0, 311.4), (674.0, 315.6), (598.0, 315.6), (530.0, 315.6), (468.0, 319.0), (458.6, 319.5),
                        (458.6, 300.0), (455.2, 290.0)],
           [0.8, 0, 0, 0, 0.8, 0.8, 0, 0, 0, 0, 0, 1.0, 0.5, 0.5],
           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], fix=(0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13)),
        # ---------------- dust cover: top 242.2 (x 490) -> 244.8 (x 690); rear roll down to the guide button; front tucks
        # into the RSB at x 703
        _o("dustcover", [(461.0, 262.0), (464.0, 257.0), (472.0, 248.0), (486.0, 242.5), (703.0, 244.2), (703.0, 263.5)],
           [0.6, 1.5, 4.0, 3.0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0.6, 1.2, 0, 0, 0], fix=(0, 4, 5)),
        # recoil-spring guide button (x 457..466, rows 259..272)
        _o("guidebtn", [(455.5, 261.0), (461.0, 258.6), (466.0, 262.0), (466.0, 272.0), (455.5, 272.0)],
           [1.2, 1.0, 0, 0, 0.5], [1, 0, 0, 0, 0], fix=(2, 3, 4)),
        # ---------------- rear sight block: rear face x 702.5 (rows 241..244), slider housing x 707..725 up to the leaf,
        # floor 243.8 under the leaf (x 725.8..756.9), front tower x 757..774 top 235.3, front face x 774.1
        _o("rsb", [(702.4, 244.3), (702.6, 241.0), (707.0, 240.0), (725.6, 241.6), (725.8, 243.8), (756.9, 243.8),
                   (757.2, 235.6), (770.4, 235.4), (774.1, 240.5), (774.1, 270.0), (702.4, 270.0)],
           [0, 0.5, 0.3, 0.3, 0.3, 0.3, 1.0, 1.5, 0.8, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0], fix=(9, 10)),
        # rear sight leaf lying down: rear tab with the notch (x 697..701, rows 231.6..237.5), top 234.3 -> 233.4, front
        # end rises to 231.5 at x 750..756 (the hinge ears), underside 241.5 over the gap
        _o("leaf", [(697.0, 231.7), (700.3, 231.7), (701.5, 234.4), (748.5, 233.4), (751.5, 231.5), (756.5, 231.6),
                    (757.0, 241.0), (726.0, 241.5), (707.0, 238.0), (701.0, 237.6), (697.4, 237.6)],
           [0.3, 0.3, 0.4, 0.6, 0.5, 0.5, 0, 0.3, 0.3, 0.3, 0.3], [1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], fix=(5, 6, 7, 8)),
        # slider on the leaf (x 707..725, top 232.0 at x 711, down to the block)
        _o("slider", [(707.0, 236.0), (709.5, 232.4), (712.5, 232.0), (716.5, 234.0), (725.5, 234.6), (725.5, 243.8),
                      (707.0, 241.0)], [0.5, 0.5, 0.5, 0.5, 0.4, 0, 0], [0, 1, 1, 0, 0, 0, 0], fix=(0, 4, 5, 6)),
        # ---------------- lower handguard (polymer): rear face x 727, bulge to 318.5 at x 739, bottom 309 -> 307.2,
        # front x 884 (retainer ring in front)
        _o("lowerhg", [(727.0, 269.5), (884.0, 269.5), (884.0, 307.2), (790.0, 309.0), (762.0, 309.5), (740.0, 318.6),
                       (728.0, 312.0)],
           [1.0, 0, 0, 10.0, 8.0, 10.0, 2.5], [0, 0, 1, 1, 1, 1, 0], fix=(0, 1, 2)),
        # ---------------- upper handguard (polymer): top 241.0 (x 785) -> 243.8 (x 880); bottom overlaps the lower one
        _o("upperhg", [(784.0, 241.0), (884.0, 243.8), (884.0, 271.0), (784.0, 271.0)], [1.5, 0, 0, 0], [1, 0, 0, 0],
           fix=(1, 2, 3)),
        # ---------------- pistol grip: top under the receiver, vertical front face by the guard (x 528, rows 318..345),
        # straight front, palm-swell back, flared toe; rear top tang under the receiver (rows 319..323)
        _o("grip", [(462.0, 318.0), (530.5, 318.0), (528.4, 330.0), (528.3, 344.5), (520.9, 346.0), (500.5, 404.0),
                    (499.0, 419.0), (489.0, 427.8), (464.0, 423.0), (455.0, 414.0), (468.5, 340.0), (469.4, 322.4),
                    (462.0, 321.0)],
           [0.0, 0.0, 1.0, 1.5, 3.0, 4.0, 4.0, 4.0, 5.0, 4.0, 6.0, 1.0, 0.5],
           [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5, 0, 0], fix=(0, 1, 11, 12)),
        # ---------------- magazine: front edge concave arc, rear edge convex arc (x 606.2 at row 324), slanted floorplate
        _o("mag", [(598.0, 312.0), (684.0, 312.0), (678.5, 320.0), (733.0, 458.0), (684.0, 495.0), (676.0, 499.5),
                   (650.0, 450.0), (606.0, 322.0)],
           [0, 0, 0, 3.0, 2.5, 3.0, 2.0, 0], [0, 0, 1, 1, 1, 1, 1, 0], [0, 0, -7.0, 0, 0, 0, 5.0, 0], fix=(0, 1)),
        # ---------------- trigger: blade x 538.8..552 at row 320 (rear edge then curving forward), tip (558.5, 345.3);
        # row crossings: 324 542.2..548.6, 330 543.0..547.0, 336 545.2..548.9, 340 548.4..552.6, 344 555.1..558.8
        _o("trigger", [(538.0, 317.0), (553.6, 317.0), (547.0, 330.0), (559.4, 344.4), (557.5, 346.2), (542.8, 330.0)],
           [0, 0, 1.5, 0.5, 0.5, 1.5], [0, 1, 1, 1, 1, 1], [0, 0, 1.2, 0, 0, -1.2], fix=(0, 1)),
        # ---------------- magazine catch: block x 584..605 under the receiver (right face 605.6 -> 603.0 at row 340),
        # bottom 345, lever x 595..599 down to row 357
        _o("magcatch", [(586.0, 315.6), (606.0, 315.6), (603.6, 338.0), (599.0, 345.5), (598.4, 349.0), (596.0, 357.4),
                        (594.0, 357.0), (595.4, 347.0), (587.0, 345.5)],
           [0, 0, 1.5, 0.8, 0.5, 0.6, 0.6, 0.5, 1.0], [0, 1, 1, 0, 0, 0, 0, 0, 0], fix=(0, 1, 3, 4, 5, 6, 7, 8)),
        # ---------------- gas block (R_gasfsb 7x grid + crossings): cup top 249.1 -> 250.0, rounded front-top corner,
        # front face x 985.8, slope to the ring top (995, 276.2), ring front x 1003.6, bayonet lug to 304.6,
        # lower body 296.9, the inner arch from the tube bottom (966.5, 268.6) to the barrel top (974, 277.8)
        _o("gasblock", [(944.0, 249.1), (982.5, 250.0), (985.8, 256.0), (985.8, 262.5), (995.0, 276.1), (1003.6, 276.4),
                        (1003.6, 304.6), (994.4, 304.6), (994.4, 296.9), (968.0, 296.9), (964.0, 293.0), (964.0, 278.2),
                        (974.0, 277.8), (966.5, 268.6), (944.0, 268.4)],
           [0, 3.0, 0.5, 3.0, 1.5, 0.6, 0.8, 0.6, 0.6, 1.0, 0.5, 0, 3.0, 2.0, 0],
           [1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0], [0, 0, 0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0],
           fix=(0, 14, 5, 6, 9, 10, 11)),
        # ---------------- front sight block: rear ring x 1058.3..1068 (top 278.2), rear lug to 305.5, tower leaning back
        # (1068, 277.3) -> (1077, 236.4), top 232.2..233.0, front face x 1092.2, front lug bottom 302 (x 1087..1092.6),
        # body bottom 295.0 between the lugs (the rod passes under it)
        _o("fsb", [(1058.3, 278.2), (1067.8, 278.2), (1068.2, 277.3), (1077.0, 236.4), (1080.0, 233.0), (1090.8, 232.2),
                   (1092.2, 235.0), (1092.6, 273.2), (1092.6, 302.0), (1086.8, 302.0), (1086.8, 295.0), (1068.6, 295.0),
                   (1068.6, 305.5), (1058.3, 305.5)],
           [0.4, 0.4, 0.6, 1.5, 1.5, 1.5, 0.6, 0, 0.6, 0.5, 0.5, 0.5, 0.6, 0.6],
           [1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], fix=(7, 8, 13, 0)),
    ])
    return T


def fitted_table():
    T = table()
    for k, o in T.items():
        R.fitted(k, o)
    return T


def overlay(out, T=None, k="R", crop=None, zoom=2):
    from PIL import Image, ImageDraw
    T = T or fitted_table()
    im = Image.fromarray(R.image(k).astype(np.uint8)).convert("RGB")
    im = im.resize((im.width * zoom, im.height * zoom), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    cols = [(255, 40, 40), (40, 200, 255), (60, 220, 60), (255, 160, 0), (220, 60, 220), (255, 255, 0), (0, 255, 160)]
    for i, (n, o) in enumerate(T.items()):
        P = o.poly_mm()
        x, r = R.b2px(P[:, 0], P[:, 1])
        pts = [((a + 0.5) * zoom, (b + 0.5) * zoom) for a, b in zip(x, r)]
        d.line(pts + pts[:1], fill=cols[i % len(cols)], width=1)
        d.text(pts[0], n, fill=cols[i % len(cols)])
    if crop:
        im = im.crop(tuple(int(c * zoom) for c in crop))
    im.save(out)


if __name__ == "__main__":
    import sys
    T = fitted_table()
    for n, o in T.items():
        print(n, o.as_list(), "rms %.2f" % (o.fit_rms or 0))
    overlay(R.os.path.join(R.HERE, "work", "inv", "outl_all.png"), T, zoom=2)
