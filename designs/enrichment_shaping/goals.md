# Pin and axial enrichment shaping

Optimize the fresh-fuel, steady-state power distribution of a finite 7 × 7 UO₂
pin lattice, with ten axial fuel segments per pin and 490 enrichment parameters.
There is no depletion or burnup history in this study.

## Agreed design choices

- Start every fuel segment at **4 wt% U-235 within uranium**.
- Allow each segment's enrichment to vary between **2 and 5 wt%**.
- Use the latest completed native reference at uniform 4 wt% enrichment:
  **k = 0.990076 ± 0.000805 (one standard error)**. The user stopped dimension
  sizing and selected power flattening at this reference k-effective.
- Keep the physical geometry, fuel density, moderator conditions and total
  power fixed during enrichment optimization.
- Preserve this reference's total U-235 mass, calculated from physical segment
  volumes and isotope number densities; an unweighted enrichment average is
  not the inventory constraint.
- Minimize the maximum segment-average fuel power density divided by the
  fuel-volume-average power density. Report radial pin powers and the axial
  profile as additional views of this three-dimensional objective.
- Use the same build123d material solids in OpenMC and diffusion. Fit material
  cross-section corrections against both the native spatial power distribution
  and k-effective, and verify proposed designs with new OpenMC calculations.
- Use evaluated isotope data, including thermal scattering for light water.
  This is an illustrative UO₂ study, not the C5G7 benchmark or the manufactured
  one-group small-lattice demonstration.

## Fixed multiplication-factor requirement

The surrogate search constrains **k = k_reference** as an equality. Native
acceptance requires **`|k - k_reference| <= 0.005`**, including two combined
standard errors from the independent reference and candidate calculations.
The target remains the initial reference value throughout all refits.

The retained baseline is
`runs/enrichment_shaping/critical_20261025T065117137080Z/size_04/baseline.json`.
Its physical dimensions are recorded in `inputs/model.toml`. Earlier sizing
evaluations and the unverified critical-size proposal remain preparation
evidence; no further geometry search is part of this study.

## Numerical implementation

The physical objective is the exact maximum local-to-average fuel power
density. The search uses its smooth log-sum-exp approximation with sharpness
50; native acceptance uses the exact segment maximum. A proposal must also
preserve U-235 mass within 1e-9 relative error and satisfy the selected k band
with the configured two-standard-error allowance. The search targets the
reference k exactly rather than using the verification band as design freedom.

All 490 enrichment parameters are independent. Fifty groups defined by the
geometric reflections share material-calibration coefficients; this reduces
the influence of native sampling noise without grouping the design parameters.
The coefficients and within-group spectra are fixed during each local search
and refreshed after an accepted native evaluation. Rejected proposals restore
the previous design and halve the next numerical enrichment step.

Continue from the latest accepted native design while feasible improvements
remain; do not stop after a fixed small number of rounds. The original uniform
reference still defines the k target and total U-235 inventory. A rejected
proposal cannot replace the accepted design or become the next starting point.
The accepted native peak/average power must decrease after every accepted update.

Retain every diffusion iterate and its feasibility. Propose the lowest-objective
feasible improvement found in that search, including a corrected feasible point
when appropriate, rather than automatically taking its last iterate. Native
verification still decides whether that proposal is accepted.

The local search stops when it has no improving feasible candidate, or native
rejections reduce the enrichment step below 0.015625 wt% after repeated retries.
This reports the limit of the current local search; it does not establish a
global optimum. Keep all physical tolerances unchanged. Automatically update
progress, constraint, power-profile and enrichment figures after each native
verification, showing rejected trials separately from the accepted history.

Each calibration supports up to 80 diffusion-only optimizer iterations before
native verification. A budget-limited candidate may proceed only if it improves
the surrogate objective and satisfies both equalities within the configured
1e-9 numerical tolerance. Such a candidate is recorded as iteration-limited;
it is not a claim of a converged surrogate optimum. Other optimizer failures
stop the search, and native acceptance remains unchanged.

A short feasibility projection may correct the final iterate before native
verification. It minimizes parameter displacement with the same physical bounds
and 1e-9 equality tolerance. Saved iterates can resume against matching native
reference inputs without repeating completed optimization steps.

The executable physical configuration and numerical criteria are recorded in
`inputs/model.toml`. Power is normalized to 1 MW for comparison. This cold,
steady-state neutronics example does not model thermal feedback or operation.
