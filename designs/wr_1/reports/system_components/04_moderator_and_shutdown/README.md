# WR-1 moderator, reactivity control and shutdown

| Source | Scope used |
| --- | --- |
| Summach, **The Commissioning of WR-1**, AECL-2552, 1966. [§§2.1, 2.3–5 and Table I](https://www.osti.gov/servlets/purl/4513855#page=6) | Physical feedback, modifications and measured startup response. |
| Tegart, **Operation of WR-1**, AECL-3523, 1969/1970. [p. 1](https://www.osti.gov/servlets/purl/4114552#page=6) | Moderator-level regulation, rapid dump and boron adjustment. |
| CNL, **WR-1 Radiological Characterization Summary**, 2020. [§2.6](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=9) | Water/helium equipment and materials. |
| CNL, **WR-1 Decommissioning EIS**, 2022. [§3.2.1.3](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=145) | Boron and purification equipment. |

WR-1 regulated reactivity through heavy-water moderator level, held by helium differential pressure. Rapid shutdown dumped moderator out of the core. Boron adjustment compensated slower changes in fuel burnup and poison conditions. These were distinct functions; the sources do not establish the boron-addition equipment as an independent fast shutdown system. [AECL-3523, p. 1](https://www.osti.gov/servlets/purl/4114552#page=6).

## Component reports

| Report | Coverage |
| --- | --- |
| [Heavy-water circulation and dump tank](heavy_water_circulation.md) | Pumps, heat exchanger, materials, dump hydraulics and thermal feedback. |
| [Helium regulation and dump valves](helium_and_dump_valves.md) | Gas equipment, level control, blower changes and measured dump response. |
| [Boron and moderator purification](boron_and_purification.md) | Addition circuits, ion exchange, collection and moderator quality. |
| [Neutron instrumentation and protective trips](neutron_control_and_trips.md) | Detection, regulation, commissioning faults and measured coefficients. |

## Coupled behaviour established during commissioning

| Trigger | Reported mechanism | Response |
| --- | --- | --- |
| Rapid moderator cooling | Upper gas space contracted; moderator level rose and added reactivity | Added moderator/process-water temperature-difference trip. |
| Excessive moderator level | Overflow entered the helium balance line and impaired dumping | Added high-level trip. |
| Excessive blower capacity | Moderator pump-up exceeded the intended reactivity addition rate | Reduced blower speed; identified altered control-valve behaviour. |
| Moderator dump | Entrained helium reached pump suction and caused gas locking | Changed dump-tank discharge routing and added baffles. |

Source: [AECL-2552, §2.1](https://www.osti.gov/servlets/purl/4513855#page=6). These observed paths justify treating gas volume, hydraulic level, cooling and control as a coupled system. They do not quantify later full-core transients.

The startup measurements also show why the system cannot be summarized as having uniformly negative feedback: the moderator temperature coefficient changed sign with temperature, and the reported B-loop coolant-void worth was positive. Core loading and coolant condition belong with every coefficient. [AECL-2552, Table I](https://www.osti.gov/servlets/purl/4513855#page=9).

The public records do not supply complete voting logic, all setpoints, shutdown-margin calculations or a lifetime safety-system qualification. Later commercial CANDU-OCR poison systems described in AECL-6436 are proposed power-station equipment and are not attributed to WR-1 here.
