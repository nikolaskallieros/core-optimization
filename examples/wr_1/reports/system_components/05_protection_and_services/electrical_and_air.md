# WR-1 normal and standby power, DC supplies, batteries and instrument air

| Source | Evidence used |
| --- | --- |
| CNL, **WR-1 Decommissioning EIS**, Rev. 4, 2022. [§3.2.2.6](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=155) | Electrical distribution, diesel generation, rectifiers, batteries and compressed-air receivers. |
| Summach, **The Commissioning of WR-1**, AECL-2552, 1966. [§2.3](https://www.osti.gov/servlets/purl/4513855#page=8) | Control-supply and electrical-interference experience. |
| CNL, **WR-1 Decommissioning EIS**, Rev. 4, 2022. [§3.2.1.4.3](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=149) | Small backup motors in the organic experimental loops. |
| CNL, **WR-1 Radiological Characterization Summary**, 2020. [Table 1](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=33) | Specific Class III supply to thermal-shield cooling pumps. |

## Electrical architecture supported by the sources

The EIS describes normal and standby AC distribution with rectified DC backed by lead-acid batteries. Four sets of diesel-driven generators in the powerhouse provided emergency or standby electrical supply. These are site-service descriptions, not evidence that four wholly independent diesel trains each supported every WR-1 essential load. [CNL 2022, p. 3-16](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=156).

| Supply or equipment | Documented role | Missing performance information |
| --- | --- | --- |
| Class IV AC | Normal supply to distribution and rectifiers | Voltage levels, feeders, capacity and loss criteria. |
| Class III AC | Standby source that could supply the rectifiers | Transfer sequence and supported load schedule. |
| Diesel generator sets | Four sets in the powerhouse | Individual ratings, shared auxiliaries, starting reliability and fuel endurance. |
| Rectifiers | Supplied DC from normal or standby AC | Output ratings, redundancy and load allocation. |
| Lead-acid battery banks | Backed DC when AC to the rectifiers was lost | Original number/configuration, discharge duty and autonomy. |
| Loop backup motors | Small motors backed main motors on fast-neutron-loop circulation pumps | Supply class, torque and achieved flow. |

Sources: [CNL 2022, pp. 3-9 and 3-16](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=149). The statement that only one battery bank remained belongs to the decommissioning description and must not be recast as the operating-era arrangement.

## Instrument and compressed air

One explicitly identified Class III load was the thermal-shield cooling pump motors. Their system used one running and one standby pump. This establishes that particular load connection; it does not fill the missing complete load schedule. [CNL 2020, Table 1](https://iaac-aeic.gc.ca/050/documents/p80124/146112E.pdf#page=33).

Compressed air arrived from the powerhouse through an underground carbon-steel main with a protective polyethylene coating. Two receiver tanks in the reactor building held reserve air. The instrument-air receiver also supplied mask air through pressure reduction and humidification; a second reducing station was available if the first could not supply the required volume. [CNL 2022, pp. 3-15–3-16](https://iaac-aeic.gc.ca/050/documents/p80124/146078E.pdf#page=155).

The public passage does not list which control or isolation valves used air, how service and instrument-air consumers were separated, or what happened to each actuator on air loss. The receiver inventory cannot therefore be translated into a shutdown hold time or valve-failure map. Mask-air and actuator duties also should not be assumed to share identical quality requirements merely because the source identifies a connection.

## Observed electrical issues

Commissioning found that some amplifiers and power supplies worked on the bench but failed in warmer equipment cabinets. Better components and heat sinks corrected the reported problem. Crane-control wiring also induced false neutron-rate trips, and annunciator circuitry suffered interference until filtering and rerouting reduced it. These were equipment-quality and installation issues, distinct from loss of bulk station power. [AECL-2552, §2.3](https://www.osti.gov/servlets/purl/4513855#page=8).

## System implications and limits

Inference: a credible essential-services model would link each load to its actual supply, transfer delay, stored-energy provision and cooling/air dependency. The available description identifies categories but not enough detail to construct that map. Flywheel rundown belongs in such a model separately from electrically driven standby operation.

No full single-line diagram, motor-control schedule, battery discharge calculation, diesel load sequence, air receiver volumes, air pressures or actuator consumption rates was recovered. Accordingly, the reports do not claim unlimited DC continuity, instantaneous transfer or a quantified common-cause failure probability. The main conclusion is that multiple supply forms existed, with the capacity and independence of complete functional paths still requiring primary electrical and pneumatic documentation.
