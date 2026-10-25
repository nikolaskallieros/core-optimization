# WR-1 pump rundown, emergency organic injection and standby cooling

| Source | Evidence used |
| --- | --- |
| Summach, **The Commissioning of WR-1**, AECL-2552, 1966. [§2.2](https://www.osti.gov/servlets/purl/4513855#page=7) | Main-pump flywheels. |
| CNL, **WR-1 Radiological Characterization Summary**, 2020. [§2.5 and Table 4](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=53) | Standby-cooler equipment and C-circuit differences. |
| CNL, **WR-1 Decommissioning EIS**, Rev. 4, 2022. [§3.2.1.4.3](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=149), [§3.2.2.5](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=155) | Organic-loop rundown and emergency injection connections. |

## Cooling functions

The reviewed sources describe three different provisions: stored rotational energy maintaining flow after drive loss, pressurized fresh coolant supplying inventory after a breach, and standby heat removal. These act on different parts of the cooling problem. Moderator dumping reduces fission power, while the cooling equipment must still handle stored heat and continuing decay heat.

| Provision | Documented equipment | Role |
| --- | --- | --- |
| Main-pump rundown | One-ton flywheels on the early main pumps | Continue circulation as rotational speed declines after drive loss. |
| Fast-neutron-loop rundown | Flywheels on loop circulation pumps and small backup motors | Loop-specific continued circulation. |
| Emergency injection | Two pressurized tanks containing fresh HB-40/OS-84 | Supply coolant through designated inlet/outlet connections. |
| Standby organic cooling | Auxiliary heat exchangers associated with degassing equipment | Remove heat under standby conditions. |
| Standby water | Essential cooling-water branch | Heat rejection for equipment unable to tolerate ordinary process-water loss. |

Sources: [AECL-2552, p. 3](https://www.osti.gov/servlets/purl/4513855#page=7); [CNL 2022, p. 3-9](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=149); [CNL 2020, §§2.5, 2.10](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=9).

## Injection routing

| Tank | Connection stated in the EIS |
| --- | --- |
| PO-TK1 | Inlet headers of primary circuits A, B and C, and experimental-loop tank/feeders. |
| PO-TK3 | A/B/C outlet feeders through the activity-monitoring supply lines. |

Source: [CNL 2022, §3.2.2.5](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=155). The fast-neutron-loop description also states connections at both inlet and outlet feeders. This is a documented two-ended delivery topology, not a reproduced accident calculation.

The shared use of activity-monitoring supply lines is an interface worth preserving in a system model. It does not establish that injection and sampling operated simultaneously or that one could impair the other; those questions require the valve and check-device arrangement.

## Standby-cooler configuration

The later inventory gives A/B standby coolers as 10.75-in OD × 96 in and the C unit as 10.75-in OD × 80 in. C's unit also cooled particulate-removal flow, whereas A/B had separate dedicated treatment coolers. External size is not cooling duty. The sources do not state whether concurrent duties, alternate alignments or dedicated reserve capacity controlled C's availability. [CNL 2020, Table 4 and §2.5](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=53).

## Performance evidence and limits

Flywheel mass alone does not define rotational inertia, and inertia alone does not give channel flow without pump and system curves. Similarly, a pressurized tank's presence does not establish delivery duration without gas pressure, usable liquid volume, downstream resistance and break conditions. No values are inferred for those missing quantities.

The public descriptions do not provide the injection tank sizes, initial pressures, initiating logic, valve failure positions, rundown curves, acceptance temperatures or a complete loss-of-coolant analysis. Commercial CANDU-OCR emergency-water backup and recirculation proposals in other sections of AECL-6436 are not imported into WR-1's equipment list. The retained evidence supports the historical provisions and connections, while quantitative adequacy remains outside this report's demonstrated scope.
