# Collections & crafting: the loot people chase

Valuables stop being "sell fodder" when they belong to **sets**. Every collectible has a set, a number, a rarity and
one line of lore. Completing a set unlocks a bonus and a display on the **Collector's Wall** in the Cortlandt stash room
(the wall shows empty outlines until you find each piece, so every raid has a "maybe today" item).
All names are invented. Data for the game: `props/collections.json`.

## Sets

| Set | Pieces | Mesh (one, many skins) | Set bonus | Where it drops |
|---|---|---|---|---|
| **Token Box**: transit tokens 1953–2003 | 8 + 1 secret | `SM_Token` (Ø23 mm coin) | Trader pays +5 % for valuables | Station booths, turnstiles, couch cushions |
| **Stoopball Cards**: '87 borough all-stars | 12 + 1 holo | `SM_TradingCard` in `SM_Toploader` | +1 stash row | Kids' rooms, bodega counters |
| **Snow Globes**: "I Survived the City" souvenirs | 6 | `SM_SnowGlobe` (scene card inside) | Galley buff lasts 25 % longer | Souvenir shops, apartments |
| **Crate Diggers**: LPs from dead labels | 8 | `SM_Record` + `SM_RecordSleeve` | Unlocks the hub record player (music in the Commons) | Walk-up apartments, record store |
| **Last Photos**: polaroids with notes | 10 | `SM_Polaroid` | Reveals a hidden stash marker per photo (quest chain) | On bodies, fridges, notice boards |
| **Mascots**: bobbleheads of the city's teams | 6 | `SM_Bobblehead` (head on a spring part) | Morale: +5 % XP in the hub-adjacent raids | Sports bars, offices |

**Rarity weights per piece**: common 55 · uncommon 30 · rare 12 · legendary 3 (the secret token, the holo card).
**Duplicates** sell, or trade 3 → 1 random missing piece at the black market.

## Crafting materials that do something
Each material is in at least two recipes. Recipes are data; the workshop UI reads `props/collections.json`.

| Material | Mesh | Used for |
|---|---|---|
| Duct tape (silver / black / camo / orange / white) | `SM_DuctTape` | Splint (+stick), barricade patch, weapon grip wrap, armor repair (+plate) |
| Paracord hank (5 colours) | `SM_Paracord` | Tripwire alarm (+cans), sling, snare, tourniquet (+stick) |
| AA battery 4-pack | `SM_Batteries` | Flashlight, radio, NVG, keycard reader bypass (+wires) |
| Superglue tube | `SM_Superglue` | Wound seal (+bandage), mod repair, lock jam |
| 3D printer filament (existing rows) | `SM_FilamentSpool` | Printed mags, sights, suppressor baffles (+steel wool) |
| Scrap metal | `SM_ScrapMetal` | Armor plate, barricade, knife |

## Built so far (3dlab)
`SM_Token` (8 skins + secret), `SM_TradingCard` + `SM_Toploader` (12 cards + holo), `SM_DuctTape` (5 skins), `SM_Batteries` (3 brands).
