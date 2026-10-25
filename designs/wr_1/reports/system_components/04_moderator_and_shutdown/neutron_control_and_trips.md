# WR-1 neutron instrumentation, regulating circuits and protective trips

| Source | Evidence used |
| --- | --- |
| Summach, **The Commissioning of WR-1**, AECL-2552, 1966. [§§2.1, 2.3–5; Table I](https://www.osti.gov/servlets/purl/4513855#page=8) | Commissioning logic changes, instrument faults and measured coefficients. |
| Tegart, **Operation of WR-1**, AECL-3523, 1969/1970. [§§1, 10](https://www.osti.gov/servlets/purl/4114552#page=6) | Regulation principle and responsibility for experimental configurations. |
| CNL, **WR-1 Decommissioning EIS**, Rev. 4, 2022. [§3.2.2.6](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=156) | Ion chambers, amplifiers, signal cables and power supplies. |

## Measurement and actuation

The neutron-power system comprised ion chambers, amplifiers, signal cables, power supplies and instrumentation. Regulation acted through helium pressure and moderator level. During commissioning, neutron-power control operated satisfactorily, whereas the thermal-power computers had unresolved drift, noise, reliability and accuracy problems. The source does not support describing both modes as equivalently demonstrated. [AECL-2552, §§2.3, 5](https://www.osti.gov/servlets/purl/4513855#page=8).

The protective system underwent daily tests for two months before startup, according to the commissioning account. This establishes the reported test programme, not a modern probabilistic reliability estimate or proof that all later configurations had identical protection.

## Faults and corrective changes

| Observed problem | Reported action |
| --- | --- |
| Crane-control wiring coupled into ion-chamber cables and caused false rate trips | Rerouted the chamber cables and installed conduit. |
| Pulse-operated annunciators responded to interference | Added filters at annunciators and, in some cases, interference sources; some false indications remained under correction. |
| Amplifiers/power supplies failed in warm cabinets despite passing bench tests | Improved component quality and added heat sinks. |
| Chamber output exceeded prediction because cavity flux gradients were smaller | Withdrew chambers for the current core; larger cadmium shutters were proposed as the permanent change before further core expansion. |
| Narrow radiation streams from chamber housings | Up to 300 mR/h reported during high-power commissioning; the cited account does not provide a complete subsequent shielding resolution. |

Source: [AECL-2552, §§2.3–5](https://www.osti.gov/servlets/purl/4513855#page=8). Proposed changes are distinguished from measures reported as performed.

## Trips added during physical commissioning

High moderator level could flood the helium balance line and slow dumping, so a high-level trip was installed. Rapid moderator cooling could raise the liquid level through upper-gas contraction; an additional moderator/process-water temperature-difference trip addressed that mechanism. These are specific recovered protective functions, not the complete plant trip matrix. [AECL-2552, §2.1](https://www.osti.gov/servlets/purl/4513855#page=6).

## Measured reactor response

| Quantity in the startup study | Reported measurement | Qualification |
| --- | ---: | --- |
| Moderator temperature coefficient at 10°C | +0.135 mk/°C | Sign depended on moderator state. |
| Same at 50°C / 80°C | 0 / −0.09 mk/°C | Not one constant across temperature. |
| Power coefficient | −0.046 mk/% | Less negative than the −0.098-mk/% design prediction. |
| Coolant temperature coefficient | −0.014 mk/°C | Dated startup result. |
| B-loop coolant void quantity | +22.5 mk | Source reports mk, without a per-percent-void denominator. |
| Moderator dump in first second | 15.5 mk removal magnitude | Not full shutdown margin or a lifetime acceptance bound. |

Source: [AECL-2552, Table I](https://www.osti.gov/servlets/purl/4513855#page=9). The commissioning programme used 19- and 25-site cores. The authors attributed part of the power-coefficient discrepancy to coolant displaced by hotter expanding sheaths, adding a positive reactivity contribution overlooked in the original prediction. [AECL-2552, §5](https://www.osti.gov/servlets/purl/4513855#page=10).

## Applicability and gaps

The evidence shows interactions among instrument location, wiring, electronics temperature, moderator hydraulics and core physics. It does not establish the final trip setpoints, channel voting, testing intervals after commissioning, detector range changes, logic independence or later-core shutdown margins. Experiment-specific hazards also required reactor technology and operations review; the early operating account stresses the practical responsibility associated with changing research loads. [AECL-3523, §10](https://www.osti.gov/servlets/purl/4114552#page=16).
