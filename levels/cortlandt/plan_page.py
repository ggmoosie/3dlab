"""plan_page.py - to-scale HTML plan of Cortlandt Station for approval (level-factory stage 2).
usage: python3 levels/cortlandt/plan_page.py -> levels/cortlandt/review/CortlandtPlan.html
Every drawing is generated from plan.py (spec metres); colours are CSS tokens so the page works in both themes."""
import os, sys, html
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan as P

ROOM_TOK = {"gate": "warm", "watch_post": "alarm", "mechanic": "warm", "trade_post": "gold", "gunsmith": "alarm",
            "workshop": "cool", "galley": "warm", "infirmary": "cool", "stash": "cool", "bunks": "dim",
            "black_market": "uv", "staff_hall": "dim", "generator": "gold"}
ROOM_NOW = {"gate": "Street stairs land here; defensive airlock", "watch_post": "Sandbagged post, CCTV wall, alarm",
            "mechanic": "Gear, keys, crafting parts", "trade_post": "Trader NPC: buy / sell", "gunsmith": "Weapons, mods, repair",
            "workshop": "Crafting bench, 3D printer", "galley": "Food buffs, social corner", "infirmary": "Medic NPC, heal",
            "stash": "Lockers, recovery locker", "bunks": "Respawn (4 bunks)", "black_market": "Quest unlock behind NO ENTRY",
            "staff_hall": "Staff corridor to the back rooms", "generator": "Powers the hub; ceiling cable runs"}


class Svg:
    """y-up world metres -> SVG user units (1 m = k), north (spec +y) up"""
    def __init__(s, x0, y0, x1, y1, k=10.0, pad=3.0):
        s.x0, s.y1, s.k, s.pad = x0 - pad, y1 + pad, k, pad
        s.W, s.H = (x1 - x0 + 2 * pad) * k, (y1 - y0 + 2 * pad) * k
        s.o = []

    def p(s, x, y):
        return (x - s.x0) * s.k, (s.y1 - y) * s.k

    def rect(s, r, cls, extra=""):
        (ax, ay), (bx, by) = s.p(r[0], r[3]), s.p(r[2], r[1])
        s.o.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bx - ax:.1f}" height="{by - ay:.1f}" class="{cls}" {extra}/>')

    def line(s, a, b, cls):
        (ax, ay), (bx, by) = s.p(*a), s.p(*b)
        s.o.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" class="{cls}"/>')

    def pl(s, pts, cls):
        q = " ".join("%.1f,%.1f" % s.p(*a) for a in pts)
        s.o.append(f'<polyline points="{q}" class="{cls}"/>')

    def circ(s, x, y, r, cls):
        cx, cy = s.p(x, y)
        s.o.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r * s.k:.1f}" class="{cls}"/>')

    def text(s, x, y, t, cls="lbl", anchor="middle"):
        cx, cy = s.p(x, y)
        for i, line in enumerate(t.split("\n")):
            s.o.append(f'<text x="{cx:.1f}" y="{cy + i * 1.15 * 11:.1f}" class="{cls}" text-anchor="{anchor}">{html.escape(line)}</text>')

    def stair(s, r, d):
        s.rect(r, "stair")
        x0, y0, x1, y1 = r
        n = int((x1 - x0) / P.TREAD)
        for i in range(0, n, 2):
            x = x0 + i * P.TREAD
            s.line((x, y0), (x, y1), "tread")
        ym = (y0 + y1) / 2
        a, b = (x0 + 0.6, ym), (x1 - 0.6, ym)
        if d < 0:
            a, b = b, a
        s.line(a, b, "arrow")

    def scale(s, x, y, m=10):
        s.line((x, y), (x + m, y), "scalebar")
        for t in (0, m):
            s.line((x + t, y - 0.4), (x + t, y + 0.4), "scalebar")
        s.text(x + m / 2, y + 0.9, f"{m} m", "dim")

    def svg(s, label):
        return (f'<svg viewBox="0 0 {s.W:.0f} {s.H:.0f}" role="img" aria-label="{html.escape(label)}" '
                f'style="width:100%;min-width:{min(int(s.W * 0.9), 1100)}px;height:auto">' + "".join(s.o) + "</svg>")


def mezz():
    s = Svg(*P.MEZZ, k=12)
    s.rect(P.MEZZ, "slab")
    for k, r in P.MEZZ_ROOMS.items():
        s.rect(r["rect"], f"room r-{ROOM_TOK[k]}")
    s.rect(P.CONCOURSE, "concourse")
    s.rect(P.NODE, "node")
    s.circ(*P.COMPASS_ROSE["center"], P.COMPASS_ROSE["r"], "rose")
    for i in range(8):
        import math
        a = i * math.pi / 4
        c = P.COMPASS_ROSE["center"]; rr = P.COMPASS_ROSE["r"] * (1 if i % 2 == 0 else 0.55)
        s.line(c, (c[0] + rr * math.cos(a), c[1] + rr * math.sin(a)), "roseline")
    for f in P.SHOPFRONTS:
        s.line((f["x0"], f["y"]), (f["x1"], f["y"]), "glass")
    for (x, y) in P.CENTER_KIOSKS:
        s.rect((x - 1.2, y - 0.8, x + 1.2, y + 0.8), "kiosk")
    for (x, y) in P.CONCOURSE_COLUMNS:
        s.rect((x - 0.18, y - 0.18, x + 0.18, y + 0.18), "col")
    for (x, y) in P.TURNSTILES:
        s.rect((x - 0.5, y - 0.12, x + 0.5, y + 0.12), "turn")
    s.line((P.FARE_LINE_X, -4), (P.FARE_LINE_X, 4), "fare")
    for st, r in zip(P.PLATFORM_STAIRS, P.PLATFORM_STAIR_RECTS):
        s.stair(r, st["dir"][0])
        s.text((r[0] + r[2]) / 2, r[3] + 0.7, f'{st["name"].replace("_", " ")} down, {st["width"]:.1f} m', "dim")
    for st, r in zip(P.STREET_STAIRS, P.STREET_STAIR_RECTS):
        s.stair(r, -1)
        s.circ(r[0] + 0.3, (r[1] + r[3]) / 2, 0.45, "globe-" + st["globe"])
        s.text((r[0] + r[2]) / 2, r[3] + 0.7 if st["globe"] == "green" else r[1] - 1.4, "from the street" if st["state"] == "open" else "welded shut", "dim")
    s.line((P.RAID_BOARD["x0"], 11.7), (P.RAID_BOARD["x1"], 11.7), "raid")
    s.text(2.0, 10.6, "RAID BOARD", "lbl strong")
    for k, r in P.MEZZ_ROOMS.items():
        x0, y0, x1, y1 = r["rect"]
        name = k.replace("_", " ").title()
        if k == "watch_post":
            s.text((x0 + x1) / 2, y0 - 0.9, name, "lbl")
            continue
        yy = (y0 + y1) / 2 + (1.2 if y0 >= 4 else -0.6) if k not in ("gate", "staff_hall") else (y0 + y1) / 2 + (6 if k == "gate" else 0)
        s.text((x0 + x1) / 2, yy, name, "lbl strong")
        if k not in ("staff_hall",):
            s.text((x0 + x1) / 2, yy - 1.2, f"{x1 - x0:.0f} × {y1 - y0:.0f} m", "dim")
    s.text(2.0, -7.8 - P.COMPASS_ROSE["r"] - 1.2, "THE NODE", "lbl strong")
    s.text(-14.0, 0.0, "MARKET", "dim")
    s.scale(-41, -14.2)
    return s.svg("Mezzanine plan")


def platform():
    s = Svg(-86, -9.5, 86, 9.5, k=7)
    s.rect((-86, -9.5, 86, 9.5), "tunnel")
    s.rect((-P.STATION_HALF_LEN, -9.5, P.STATION_HALF_LEN, 9.5), "slab")
    s.rect(P.TRACK_A, "track"); s.rect(P.TRACK_B, "track")
    s.rect(P.PLATFORM, "platform")
    for e in (P.PLATFORM[1], P.PLATFORM[3]):
        s.line((P.PLATFORM[0], e), (P.PLATFORM[2], e), "edge")
    for t in P.TRAINS:
        s.rect(t["rect"], "train" + (" derelict" if "Derelict" in t["kit"] else ""))
    for z, r in P.PLATFORM_ZONES.items():
        s.rect(r["rect"], "zone")
        s.text((r["rect"][0] + r["rect"][2]) / 2, (r["rect"][1] + r["rect"][3]) / 2 - 0.3, z.replace("_", " "), "dim")
    for (x, y) in P.PLATFORM_COLUMNS:
        s.rect((x - 0.15, y - 0.15, x + 0.15, y + 0.15), "col")
    for st, r in zip(P.PLATFORM_STAIRS, P.PLATFORM_STAIR_RECTS):
        s.stair(r, st["dir"][0])
    for t in P.RANGE_TARGETS:
        s.rect((t["x"] - 0.15, 5.2, t["x"] + 0.15, 8.3), "target")
        s.text(t["x"], 9.0, f'{t["dist"]} m', "dim")
    s.text(-60, -7.2, "TRACK A  · live train, 4 cars", "lbl", "middle")
    s.text(-60, 6.3, "TRACK B  · 3 derelict cars", "lbl", "middle")
    s.text(-80, -1, "tunnel fades to black", "dim", "start")
    s.scale(-84, -11.2 + 1.5)
    return s.svg("Platform plan")


def street():
    s = Svg(*P.STREET_BOUNDS, k=4.2)
    s.rect(P.STREET_BOUNDS, "lot")
    s.rect((-100, -P.AVENUE_HALF - P.SIDEWALK_AVE, 100, P.AVENUE_HALF + P.SIDEWALK_AVE), "sidewalk")
    s.rect((-100, -P.AVENUE_HALF, 100, P.AVENUE_HALF), "road")
    for c in P.CROSS_STREETS:
        s.rect((c["x"] - P.CROSS_HALF - P.SIDEWALK_CROSS, -62, c["x"] + P.CROSS_HALF + P.SIDEWALK_CROSS, 62), "sidewalk")
        s.rect((c["x"] - P.CROSS_HALF, -62, c["x"] + P.CROSS_HALF, 62), "road")
        s.text(c["x"] + 0.0, 52, c["name"], "lbl strong")
    s.text(20, -0.8, "GREENWICH ST", "lbl strong")
    for b in P.STREET_BLOCKS:
        s.rect(b, "bldg")
    pl = P.STREET_ZONES["plaza"]["rect"]
    s.rect(pl, "plaza"); s.text((pl[0] + pl[2]) / 2, (pl[1] + pl[3]) / 2, "FEMA camp\n(pocket plaza)", "lbl")
    ph = P.STREET_ZONES["pharmacy"]["rect"]
    s.rect(ph, "hero"); s.text((ph[0] + ph[2]) / 2, (ph[1] + ph[3]) / 2 + 1, "Cortlandt\nDrugs", "lbl")
    for w in P.STREET_ZONES["walkups"]["rects"]:
        s.rect(w, "hero"); s.text((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, "walk-up", "dim")
    sp = P.STREET_ZONES["spawn"]["rect"]
    s.rect(sp, "spawn"); s.text((sp[0] + sp[2]) / 2, sp[3] + 1.5, "START", "lbl strong")
    s.pl(P.TUTORIAL_PATH, "path")
    s.rect(P.BARRICADE, "barricade")
    for st, r in zip(P.STREET_STAIRS, P.STREET_STAIR_RECTS):
        s.rect(r, "stair-street globe-" + st["globe"])
    s.text(-38, 13.5, "green stair\n(open)", "dim"); s.text(-38, -14, "red stair\n(welded)", "dim")
    for (x, y) in P.ZOMBIE_SPAWNERS:
        s.circ(x, y, 1.0, "zspawn")
    s.scale(-98, -60.5, 20)
    return s.svg("Street plan")


def section():
    s = Svg(-50, -1.6, 50, 11.5, k=11, pad=2)
    s.rect((-50, 10.0, 50, 10.4), "slabcut")                      # street slab
    s.rect((-42, 8.8, 42, 9.2), "slabcut")                        # mezz ceiling slab
    s.rect((-42, 4.8, 42, 5.2), "slabcut")                        # mezz floor slab
    s.rect((-45, -0.4, 45, 0.0), "slabcut")                       # platform
    s.rect((-45, -1.5, 45, -1.1), "slabcut")                      # trackbed
    for z0, z1, lab in ((5.2, 8.8, "Mezzanine · 3.6 m clear"), (0.0, 4.3, "Platform · 4.3 m to ceiling")):
        s.text(-30, (z0 + z1) / 2, lab, "lbl strong", "start")
    s.text(-49, 10.9, "Street · sidewalk z 10.0", "lbl strong", "start")
    for st, r in zip(P.STREET_STAIRS[:1], P.STREET_STAIR_RECTS[:1]):
        s.pl([(r[0], 10.0), (r[2], 5.2)], "stairline")
    for st, r in zip(P.PLATFORM_STAIRS, P.PLATFORM_STAIR_RECTS):
        a, b = (r[0], 5.2), (r[2], 0.0)
        if st["dir"][0] < 0:
            a, b = (r[2], 5.2), (r[0], 0.0)
        s.pl([a, b], "stairline")
    s.text(-37.5, 7.4, "27 risers\n17.8 cm", "dim"); s.text(1.5, 2.2, "29 risers\n17.9 cm", "dim"); s.text(28.5, 2.2, "east stair", "dim")
    for z in (0, 5.2, 8.8, 10.0):
        s.text(49, z + 0.15, f"z {z:.1f}", "dim", "end")
    return s.svg("Section along the platform")


def rooms_table():
    rows = []
    for k, r in P.MEZZ_ROOMS.items():
        x0, y0, x1, y1 = r["rect"]
        near = "yes" if k in P.NODE_SERVICES else ""
        rows.append(f'<tr><td><span class="sw r-{ROOM_TOK[k]}"></span>{k.replace("_", " ").title()}</td><td class="num">{x1 - x0:.0f} × {y1 - y0:.0f}</td>'
                    f'<td>{html.escape(r["was"])}</td><td>{html.escape(ROOM_NOW[k])}</td><td>{html.escape(r["light"])}</td><td>{near}</td></tr>')
    return "".join(rows)


CSS = """
/* layout: one reading column of drawings, tables below; signage bands mark each level like a station name tablet */
:root{--bg:#eef0ee;--paper:#fbfcfb;--ink:#15181a;--mute:#5d6563;--rule:#c9cfcc;--band:#15181a;--bandink:#f4f6f5;
--green:#1d8a4e;--red:#c8322b;--gold:#c99a2e;--warm:#d98b4b;--cool:#5d93b5;--dim:#7a6f8f;--uv:#7c4fc4;--alarm:#c8322b;
--slab:#d6dbd8;--conc:#f6f7f4;--track:#3b3f42;--road:#4a4f53;--side:#c4c8c3;--bldg:#a6aaa5;--lot:#dfe2de;
--disp:"Archivo","Helvetica Neue",Arial,sans-serif;--body:"Archivo","Helvetica Neue",Arial,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#101314;--paper:#171b1c;--ink:#e9eceb;--mute:#9aa3a0;--rule:#2c3335;
--band:#e9eceb;--bandink:#101314;--slab:#2a3032;--conc:#20262a;--track:#0b0c0d;--road:#2f3438;--side:#41474a;--bldg:#555b5e;--lot:#1c2123;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#101314;--paper:#171b1c;--ink:#e9eceb;--mute:#9aa3a0;--rule:#2c3335;--band:#e9eceb;--bandink:#101314;
--slab:#2a3032;--conc:#20262a;--track:#0b0c0d;--road:#2f3438;--side:#41474a;--bldg:#555b5e;--lot:#1c2123;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--body);line-height:1.5}
.wrap{max-width:1180px;margin:0 auto;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:28px 64px;display:grid;gap:28px}
header h1{font-family:var(--disp);font-weight:800;font-size:clamp(30px,5vw,52px);letter-spacing:-.01em;margin:0;text-wrap:balance}
header p{margin:6px 0 0;color:var(--mute);max-width:68ch}
.stack{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.chip{font-family:var(--mono);font-size:12.5px;padding:4px 10px;border:1px solid var(--rule);border-radius:999px;background:var(--paper);font-variant-numeric:tabular-nums}
.band{background:var(--band);color:var(--bandink);display:flex;align-items:center;gap:12px;padding:10px 14px;font-family:var(--disp);font-weight:700;letter-spacing:.04em;text-transform:uppercase;font-size:15px}
.band .bul{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-size:13px;color:#fff;flex:none}
.band small{font-family:var(--mono);text-transform:none;letter-spacing:0;font-weight:400;opacity:.8;margin-left:auto}
section{display:grid;gap:12px;min-width:0}
.fig{background:var(--paper);border:1px solid var(--rule);overflow-x:auto}
.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px 28px;margin:0;padding:0;list-style:none}
.notes li{padding-left:14px;border-left:3px solid var(--rule);color:var(--ink)}
.notes b{font-weight:700}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:760px}
th{text-align:left;font-family:var(--disp);font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mute);border-bottom:2px solid var(--ink);padding:6px 8px}
td{border-bottom:1px solid var(--rule);padding:6px 8px;vertical-align:top}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.tbl{overflow-x:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:baseline}
.dec{display:grid;gap:10px;counter-reset:d}
.dec div{background:var(--paper);border:1px solid var(--rule);padding:12px 14px;display:grid;gap:4px}
.dec b{font-family:var(--disp)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}
h2{font-family:var(--disp);font-size:20px;margin:0}
p{margin:0;max-width:72ch}
/* svg */
svg text{font-family:var(--body);font-size:11px;fill:var(--ink)}
svg .dim{font-family:var(--mono);font-size:9.5px;fill:var(--mute)}
svg .strong{font-weight:700;font-size:11.5px;letter-spacing:.03em}
.slab{fill:var(--slab)}.concourse{fill:var(--conc);stroke:var(--rule)}
.node{fill:none;stroke:var(--gold);stroke-width:2;stroke-dasharray:6 4}
.room{stroke:var(--paper);stroke-width:2;fill-opacity:.32}
.r-warm{fill:var(--warm);background:var(--warm)}.r-alarm{fill:var(--alarm);background:var(--alarm)}.r-gold{fill:var(--gold);background:var(--gold)}
.r-cool{fill:var(--cool);background:var(--cool)}.r-dim{fill:var(--dim);background:var(--dim)}.r-uv{fill:var(--uv);background:var(--uv)}
.rose{fill:none;stroke:var(--gold);stroke-width:1.5}.roseline{stroke:var(--gold);stroke-width:1}
.glass{stroke:var(--cool);stroke-width:3}.kiosk{fill:var(--paper);stroke:var(--cool);stroke-width:1.5}
.col{fill:var(--ink)}.turn{fill:var(--mute)}.fare{stroke:var(--mute);stroke-dasharray:3 3}
.stair{fill:var(--side);stroke:var(--ink);stroke-width:1}.tread{stroke:var(--mute);stroke-width:.6}
.arrow{stroke:var(--ink);stroke-width:2;marker-end:none}
.raid{stroke:var(--red);stroke-width:5}
.globe-green{fill:var(--green)}.globe-red{fill:var(--red)}
.tunnel{fill:var(--track);opacity:.55}.track{fill:var(--track)}.platform{fill:var(--conc);stroke:var(--rule)}
.edge{stroke:var(--gold);stroke-width:3}.train{fill:var(--slab);stroke:var(--ink);stroke-width:1}.derelict{fill:var(--bldg)}
.zone{fill:var(--gold);fill-opacity:.12;stroke:var(--gold);stroke-dasharray:4 3}.target{fill:var(--red)}
.lot{fill:var(--lot)}.sidewalk{fill:var(--side)}.road{fill:var(--road)}.bldg{fill:var(--bldg);stroke:var(--lot);stroke-width:1.5}
.plaza{fill:var(--green);fill-opacity:.25;stroke:var(--green)}.hero{fill:var(--gold);fill-opacity:.45;stroke:var(--gold)}
.spawn{fill:var(--cool);fill-opacity:.5}.path{fill:none;stroke:var(--cool);stroke-width:3;stroke-dasharray:8 5}
.barricade{fill:none;stroke:var(--red);stroke-width:2.5;stroke-dasharray:10 4}.stair-street{stroke:var(--ink)}
.zspawn{fill:var(--red);opacity:.7}.scalebar{stroke:var(--ink);stroke-width:1.5}
.slabcut{fill:var(--mute)}.stairline{fill:none;stroke:var(--ink);stroke-width:2.5}
"""


def page():
    z = [("Street", P.Z_STREET), ("Mezz ceiling", P.MEZZ_CEIL), ("Mezzanine", P.Z_MEZZ), ("Platform ceiling", P.PLATFORM_CEIL), ("Platform", 0.0)]
    chips = "".join(f'<span class="chip">{n} z {v:.1f} m</span>' for n, v in z)
    dec = "".join(f"<div><b>{html.escape(a)}</b><span>{html.escape(b)}</span></div>" for a, b in P.DECISIONS)
    B = P.BUDGETS
    return f"""<title>Cortlandt Station Plan</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>Cortlandt Station</h1>
  <p>Layout v3, "The Arcade". The hub level for LootNShoot, drawn to scale from <code>plan.py</code>, the one source of truth the Blender
  layout and the Unreal build both read. Three levels: the city street, a bright underground arcade (the safe hub), and the island platform where the train leaves for raids.</p>
  <div class="stack">{chips}</div>
</header>

<section>
  <div class="band"><span class="bul" style="background:var(--green)">S</span>Section along the platform<small>vertical stack, 1 : 100 scale bar</small></div>
  <div class="fig">{section()}</div>
</section>

<section>
  <div class="band"><span class="bul" style="background:var(--gold)">M</span>Mezzanine · the safe hub<small>84 × 24 m, z 5.2, clear 3.6 m</small></div>
  <div class="fig">{mezz()}</div>
  <ul class="notes">
    <li><b>Flow.</b> Street stairs land in the Gate, past the watch post and the turnstiles into the Market, then the Node.</li>
    <li><b>The Node</b> holds the main stair down, the raid board on the north wall and a compass-rose floor. Trade post, infirmary, gunsmith and stash are the four bays that touch it.</li>
    <li><b>Glass shopfronts</b> both sides of an 8 m concourse, white riveted columns, four glass display kiosks down the centre line.</li>
    <li><b>Bunks</b> beside the stash: wake, gear up, reach the raid board in about 15 s.</li>
  </ul>
</section>

<section>
  <div class="band"><span class="bul" style="background:var(--cool)">P</span>Platform · the way out<small>70 × 9 m island, z 0</small></div>
  <div class="fig">{platform()}</div>
  <ul class="notes">
    <li><b>Track A</b> (south): the live train, 4 cars. Board between the two stair feet; pick a stop at the route board by the lead cab.</li>
    <li><b>Track B</b> (north): 3 derelict cars, then the test range down the tunnel with targets at 10, 25 and 50 m.</li>
    <li><b>I-beam columns</b> every 15 ft in two rows; both tunnel mouths fade to black to hide the ride loading screen.</li>
  </ul>
</section>

<section>
  <div class="band"><span class="bul" style="background:var(--red)">1</span>Street · danger zone and tutorial<small>200 × 124 m, z 10</small></div>
  <div class="fig">{street()}</div>
  <ul class="notes">
    <li><b>Tutorial path</b> (dashed blue): overrun NYPD checkpoint, Cortlandt St, Cortlandt Drugs for the first loot, the green-globe stair, down. About 60 m.</li>
    <li><b>Entrances</b> are ordinary sidewalk stairs against the building line: green globes open, red globes welded shut.</li>
    <li><b>Barricade ring</b> (red dashes) with {len(P.ZOMBIE_SPAWNERS)} zombie spawners outside; the Tide breaks in at telegraphed gaps.</li>
    <li><b>{P.STOREFRONT_COUNT} storefronts</b>, {P.INTERIOR_COUNT} enterable, {P.WALKUP_COUNT} walk-ups; a FEMA tent camp in the pocket plaza.</li>
  </ul>
</section>

<section>
  <h2>Mezzanine rooms</h2>
  <div class="tbl"><table><thead><tr><th>Room</th><th>Size m</th><th>Was</th><th>Now</th><th>Light identity</th><th>Touches the Node</th></tr></thead>
  <tbody>{rooms_table()}</tbody></table></div>
</section>

<section class="grid2">
  <div style="display:grid;gap:10px;min-width:0">
    <h2>Budgets</h2>
    <div class="tbl"><table style="min-width:0"><tbody>
      <tr><td>Whole map</td><td class="num">&lt; {B["map_tris"]:,} tris</td></tr>
      <tr><td>Unique meshes</td><td class="num">&lt; {B["unique_meshes"]}</td></tr>
      <tr><td>Material instances</td><td class="num">&lt; {B["material_instances"]}</td></tr>
      <tr><td>Kit prop LOD0</td><td class="num">{B["kit_prop_tris"][0]:,}–{B["kit_prop_tris"][1]:,} tris</td></tr>
      <tr><td>Lights per floor pixel (p95)</td><td class="num">≤ {B["lights_per_floor_px_p95"]}</td></tr>
      <tr><td>Shadow casters</td><td class="num">≈ {B["shadow_casters"]}</td></tr>
      <tr><td>Exposure</td><td class="num">EV100 {B["ev100"]}</td></tr>
    </tbody></table></div>
    <p style="color:var(--mute)">Lighting follows GunRange: hero lights per zone, every other fixture an emissive lens, Lumen fills between.</p>
  </div>
  <div style="display:grid;gap:10px;min-width:0">
    <h2>Decisions before modelling</h2>
    <div class="dec">{dec}</div>
  </div>
</section>

<section>
  <h2>Build order after approval</h2>
  <ul class="notes">
    <li><b>Textures + print atlas</b>: subway tile, green trim, terrazzo, I-beam paint, brick, asphalt, sidewalk; signage and route board on one atlas.</li>
    <li><b>Kit in Blender</b>: columns, shopfront module, kiosk, turnstile, stair + railing, globe newel lamp, train car, platform edge, street furniture. Gate: mesh_qc 0 errors.</li>
    <li><b>Layout</b> from plan.py into map.blend + layout.json (meshes, shell boxes, spawners, lights, doors); Cycles review sheet.</li>
    <li><b>Export</b> FBX per kit piece + the UE import and map-build scripts (run on your machine: this cloud box has no Unreal).</li>
  </ul>
</section>
</div>"""


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "review"), exist_ok=True)
    out = os.path.join(HERE, "review", "CortlandtPlan.html")
    open(out, "w").write(page())
    print(out)
