# WR-1 pressurizing pumps and degassing, expansion and reserve inventory

| Source | Evidence used |
| --- | --- |
| CNL, **WR-1 Radiological Characterization Summary**, Rev. 2, 2020. [§2.5](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=9), [Table 4](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=53) | Pump count, tank functions, dimensions and circuit distinctions. |
| Summach, **The Commissioning of WR-1**, AECL-2552, 1966. [§2.2](https://www.osti.gov/servlets/purl/4513855#page=7) | Gas-locking and channel-level instability during fuel changing. |
| CNL, **WR-1 Decommissioning EIS**, Rev. 4, 2022. [§§3.2.1.4.3, 3.2.2.5](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=149) | Experimental-loop auxiliary links and separate emergency tanks. |

## Roles and circuit arrangement

Each main organic circuit had a degassing system with two pressurizing pumps, a degassing tank, off-gas condensation, volatile recycle and particulate removal. The tank combined several functions: accepting thermal expansion from the primary and degassing systems, holding reserve coolant for transfers/sampling/minor leakage, and maintaining suction head for the pressurizing pumps. The word “expansion” describes a documented function of this tank; the reviewed inventory does not justify inventing an additional dedicated expansion vessel. [CNL 2020, §2.5](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=9).

| Item | A/B description | C-circuit distinction |
| --- | --- | --- |
| Degassing tank | 6-ft outside diameter × 12-ft reported length/height; carbon steel | 7-ft OD × 10.82 ft; different material entry in the source. |
| Pressurizing pumps | Two per degassing system | Two stated for the C system as well; curves and duty split absent. |
| Separate inventory tank | No corresponding A/B item in Table 4 | CDO-TK18, 54-in OD × 76 in; accepts surplus degassing-tank coolant during C-circuit fuelling. |
| Seal drainage | A/B pump-seal drain tank, 6-in diameter × 17 ft | Separate C drain and waste-drain tanks distinguish returnable drainage from seal drainage. |

Source: [CNL 2020, Table 4](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=53). Overall vessel dimensions are not working liquid volumes. The material typography in this retrospective table is not treated as a certified procurement specification.

## Operation and interfaces

Thermal expansion increases the coolant volume without requiring equivalent fresh makeup. Transfers to purification, sampling losses and small leaks instead alter available inventory. The degassing tank accommodates both effects, while its liquid level also affects pump suction conditions. These distinctions follow from the source's stated functions; no level-control law or minimum suction margin is reconstructed.

The organic experimental loops drew and returned small flows through the A/B degassing arrangements. That connection supplied conditioned coolant and pressure support while the loop's own circulation pumps drove test-section flow. It means a loop can have separate main circulation without being chemically isolated from the A/B auxiliary system. [CNL 2022, p. 3-9](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=149).

The two fresh-coolant emergency-injection tanks were separate equipment with inlet/outlet connections. Reserve coolant in the degassing system must not automatically be credited as the same protected emergency inventory. [CNL 2022, §3.2.2.5](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=155).

## Experience and limitations

During commissioning, opening a fuel channel admitted air, partly gas-locked pumps and produced oscillating channel levels, sometimes with overflow. Piping changes were required. The observation supports a coupling between open-channel conditions, gas entrainment and inventory stability; it does not identify a unique failed pressurizing pump. [AECL-2552, §2.2](https://www.osti.gov/servlets/purl/4513855#page=7).

Pump head/flow curves, operating pressure, tank level bands, gas-space pressure control, relief settings, usable reserve volume and switchover logic were not recovered. Nor is two installed pumps proof that either alone met every duty. A transient inventory calculation would need these values plus circuit thermal expansion and connected loop volumes. External dimensions alone cannot close that balance.
