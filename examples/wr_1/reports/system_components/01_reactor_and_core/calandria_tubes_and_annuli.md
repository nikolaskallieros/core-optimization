# WR-1 aluminum calandria tubes and CO₂ annuli

| Source | Evidence used |
| --- | --- |
| Robertson, **Status Report on the WR-1 Reactor**, 65-CNA-205, 1965. [Fig. 2](https://www.osti.gov/servlets/purl/4891392#page=13) | Initial tube bores, walls, materials and lattice. |
| Tegart, **Operation of the WR-1 Organic Cooled Research Reactor**, AECL-3523, 1969/1970. [§8, p. 10](https://www.osti.gov/servlets/purl/4114552#page=15) | CO₂-space leak monitoring and a channel leak. |
| CNL, **WR-1 Reactor Radiological Characterization Summary**, Rev. 2, 2020. [§2.2](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=8) | Later tube count and diameters. |
| CNL, **WR-1 Decommissioning EIS**, Rev. 4, 2022. [§3.2.1.1](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=145) | Annulus purpose and separation of fluids. |

## Arrangement

Each calandria tube separated the surrounding heavy water from the concentric pressure channel. The intervening annular space contained CO₂. The fuel and upward-flowing organic were inside the pressure tube, so the annulus was an insulating and monitored space rather than a normal fuel-cooling passage. [CNL 2022, p. 3-5](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=145).

From inside outward, the documented order is fuel bundle, organic coolant, pressure-tube wall, CO₂ annulus, aluminum calandria-tube wall and heavy-water moderator. Heat could cross these layers, but the liquids had separate boundaries. This distinction matters when interpreting a leak indication: organic points toward the channel side, whereas heavy water points toward the moderator-side boundary. The early operating report explicitly describes monitoring for both fluids. [AECL-3523, §8](https://www.osti.gov/servlets/purl/4114552#page=15).

## Dated geometry

| Configuration | Calandria tubes | Bore and wall |
| --- | --- | --- |
| April 1965 design | 42 small and 13 large; aluminum identified as Al 57 S in the drawing | Fig. 2: 3.695/4.755 in ID and 0.050/0.060 in wall, respectively. |
| Metric conversion of drawing | Same initial configuration | Small: 93.9 mm ID, 1.27 mm wall; large: 120.8 mm ID, 1.52 mm wall. |
| 2020 retrospective inventory | 30 large and 24 small, total 54 | 121 mm large ID and 99 mm small ID; this is not the same small bore as the initial drawing. |

Sources: [1965 Fig. 2](https://www.osti.gov/servlets/purl/4891392#page=13); [CNL 2020, p. 8 and footnotes](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=8). The 1965 prose rounds the bores to 3.69 and 4.75 in. The drawing values are used for the calculations below.

## Nominal annulus calculation

For concentric tubes, radial clearance is one-half of the calandria-tube inside diameter minus the pressure-tube outside diameter. Applying only the small-bore 1965 drawing dimensions gives:

| Planned pressure tube | Inside diameter (mm) | Wall (mm) | Calculated radial clearance (mm) |
| --- | ---: | ---: | ---: |
| Stainless steel | 82.55 | 1.0414 | 4.61 |
| Zircaloy-4 or Zr–2.5Nb | 82.55 | 1.8034 | 3.85 |
| SAP-895 | 82.55 | 2.9464 | 2.71 |

Calculation: clearance = (93.853 − 82.55 − 2 × pressure-tube wall)/2. These are nominal geometric deductions, not measurements. They exclude eccentricity, spacers, creep, thermal expansion and manufacturing tolerances. They do not establish the later 99-mm-bore configuration or the coolant gap outside the fuel bundle. [Dimension basis: 1965 Fig. 2](https://www.osti.gov/servlets/purl/4891392#page=13).

## Operating experience

Tegart describes a channel connection/leak problem found through the CO₂-space monitoring. Organic fouled the surrounding calandria tube; after roughly two years the deposit appeared to have disappeared. The author proposed activated oxygen in the annulus as a cleaning mechanism, referring to laboratory observations. This is a historical interpretation of an observation, not evidence that every deposit would self-remove or that annulus gas chemistry could be ignored. [AECL-3523, p. 10](https://www.osti.gov/servlets/purl/4114552#page=15).

## Interfaces and limits

The annulus links pressure-tube integrity, calandria-tube integrity, heat leakage to the moderator and leak surveillance. Inference: changing its width or gas composition can change thermal coupling, while tube contact could create a different heat path. The reviewed reports do not quantify either effect.

No common dated set of annulus gas flow, pressure, impurity specification, spacer geometry, leak-detector thresholds or minimum remaining clearance was recovered. The later 54-tube inventory and initial 55-position lattice must remain separately labelled. A complete channel map and drawing revisions are needed before assigning one bore, material and clearance to every location.
