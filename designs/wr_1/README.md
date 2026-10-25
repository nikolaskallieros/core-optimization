# WR-1 design

This design defines WR-1 geometry, material choices, operating scenarios and
optimization goals. Native transport, inventory handling, cycle replay,
calibration, optimization and plotting are provided by the packages in `src/`.
The model declares channel locations and materials. A small geometry builder
creates the shared CAD; the library converts it to native cells and computes
live diffusion mesh fractions.
See the [project README](../../README.md) for installation.

| Location | Purpose |
| --- | --- |
| `model/case.py` | Connect a scenario to `CaseModel` and `CycleCase` |
| `build_geometry.py` | One shared geometry definition and an entry point to export the CAD |
| `model/design.py`, `graphite.py`, `sleeves.py` | WR-1 design variables and physical choices |
| `model/materials.py`, `data.py` | Fuel recipe, isotope substitutions and evaluated-data selection |
| `model/calibration.py` | Regional fitting groups, material-response bindings and observation weights |
| `inputs/study.toml` | Commented operating settings, optimization goals and six labeled scenarios |
| `inputs/model.toml` | Commented lattice, channel construction, material compositions and data choices |
| `inputs/nuclear_data/` | Two manifests identifying evaluated sources and their checksums |
| `scripts/` | Six preparation entry points, one runner and one data-preparation entry point |
| `scripts/plotting/render.py` | WR-1 geometry layers and calls to the shared figure renderers |
| `results/wr1_optimization/` | Retained figures; numerical records stay local and ignored |
| `reports/` | Public-source WR-1 system descriptions and operating experience |

## Nuclear data and running a scenario

Full WR-1 transport needs evaluated neutron and thermal-scattering data. Allow
at least 10 GiB free for preparation and scratch files. From the repository root
after installing the project:

```sh
.venv/bin/python -m designs.wr_1.scripts.prepare_nuclear_data
```

Data go to `.cache/nuclear_data/`. To use another disk, set
`CORE_OPTIMIZATION_DATA` to its data directory for preparation and all subsequent
calculations. `CORE_OPTIMIZATION_ROOT` selects a different workspace for caches
and runs; these paths otherwise resolve from the working directory.

Preparation verifies evaluated sources and uses NJOY for requested temperatures.
The shared thermal-scattering processor accepts table names, bound nuclides and
temperatures declared by the case; graphite and water use the same interface.
`get_nuclear_data.endf` retrieves selected raw ENDF members for processing, while
`get_nuclear_data.depletion` installs the processed depletion library.
The bundled depletion chain can also be unpacked on its own without a download:

```sh
.venv/bin/python -m get_nuclear_data.prepare_chain
```

To prepare and run optimization step 1:

```sh
.venv/bin/python -m designs.wr_1.scripts.prepare_step_1
.venv/bin/python -m designs.wr_1.scripts.run_optimization
.venv/bin/python -m designs.wr_1.scripts.plotting.render
```

Preparation checks the declared inputs and exports OpenMC XML under
`runs/wr_1/optimization_step_1/prepared/`; it does not run transport or require a
large data download. Set `STEP` in `run_optimization.py` to select a scenario.
The runner requires the complete nuclear-data library prepared above.

## Inputs and outputs

The information flows in one direction:

```text
inputs/study.toml + its declared model and source files
    → load_configuration(..., scenario)
    → model/case.py binds the resolved values to CaseModel
    → scripts/prepare_step_N.py or scripts/run_optimization.py
    → shared preparation, calibration and optimization in src/
    → runs/wr_1/optimization_step_N/
```

Start with [study.toml](inputs/study.toml). Its comments explain units, ring order
and the purpose of the main controls. Geometry and operating conditions come
first; each scenario has a labeled section. Calibration and search tuning are
marked as advanced settings.

| To change | Edit in `study.toml` |
| --- | --- |
| Power, active height or number of cycles | `[defaults.cycle]` |
| The common criticality goal | `[defaults.cycle.objective]` |
| Initial pitches, radii, enrichments or cycle duration | `[scenarios."N".cycle.design]` |
| A scenario's burnup or power limits | `[scenarios."N".cycle.objective]` |
| EOL acceptance or fresh-fuel mass limit | The scenario's `optimization.eol_keff` or `optimization.fresh_fuel_mass` section |
| Parameter bounds or search step sizes | `[defaults.optimization.bounds]` and `learning_rates`, with scenario overrides below |

Edit shared settings under `defaults` in `study.toml`; edit only scenario-specific
differences under `scenarios`. Objects merge recursively; lists replace whole
lists. A `$ref` such as `#/settings/cycle/objective/minimum_keff` reuses a value
after those overrides are applied. The native acceptance threshold and surrogate
target therefore follow the same declared goal, and the cycle-duration bounds
are defined once. Missing or circular references fail during loading.

`model.toml` owns the physical definitions shared by the six scenarios. Model
functions consume the resolved values without rereading files or supplying a
second set of case defaults. Preparation writes `resolved_inputs.json`, `case.json`,
`geometry.step`, OpenMC XML and readable resolved inputs into `prepared/`. These are generated records of the
inputs used; edit the source TOML to change a case. Inputs changed after loading
require a reload, and a different scenario or changed sources require a new run.

The default surrogate uses the same physical regions, dimensions and boundary
conditions as the native OpenMC model. `build_geometry.py` builds the material
solids from the resolved design. The shared OpenMC adapter converts those actual
solids through STEP and GEOUNED, with FreeCAD isolated in `.geouned`.
The WR-1 diffusion adapter retains the live analytical parameterization of those
same solids: Gmsh meshes the sector face, and exact circle/polygon intersections
preserve derivatives of channel positions, radii and sleeve thicknesses.
The prepared STEP contains actual build123d solids, material identities and
boundary faces. Saved STEP inputs are fixed; optimization uses live parameters.

To export just the CAD for inspection, set `STEP` in `build_geometry.py` and run:

```sh
.venv/bin/python -m designs.wr_1.build_geometry
```

Preparation and calibration select the resulting `DiffusionGeometry` through
`CaseModel`. Calibration receives that mapping and changes material
coefficients, without creating a separate geometry. A different physical layout
requires an explicit `surrogate_mesh` override and a new fit. Both mappings retain
the same region labels and represented material volumes.

New calculations write to `runs/wr_1/optimization_step_*/results/`, preserving the
historical results. In the plotting script, `STEPS` selects scenarios and
`USE_RETAINED_RESULTS` selects historical records or new runs. Figures are
regenerated under `runs/wr_1/figures/`. The common renderers use the retained
numerical values; the original published figure files remain unchanged.

Each historical step has `data/` for numerical histories, fits and optimization
records, `states/` for OpenMC reference states, and `figures/` for plots. JSON
metadata and their `.npz` arrays form a pair and must remain together locally.
Only the figures are included in Git; a fresh clone must run a scenario before
it can regenerate numerical plots. Small regression cases and their inputs live
in `tests/fixtures/`.

The two evaluated-data manifests retain source URLs and checksums. Processing
engines live in `get_nuclear_data`; the design supplies its required materials,
temperatures and source selection. NJOY decks and processing receipts are
generated under the ignored nuclear-data cache, rather than stored as inputs.
The manifests remain JSON verification records and do not need routine editing.
Additional thermal tables are declared under
`transport.nuclear_data.extensions` in `model.toml`; the shared processor updates
each table and its bound nuclide in a separate library directory.

## Reproducibility

Historical fits and acceptance decisions describe the original runs. The
refactored code does not revalidate those calculations or silently resume their
checkpoints. New runs compare actual input values, scenario and geometry policy.
Changes to model or solver behavior require a new run directory and calibration;
source-code hashes do not control execution.

Three optional regressions require local historical results and skip in a fresh
clone. The installation and numerical checks use small independent inputs and
do not resume a production search or establish acceptance of a reactor design.

Historical step-one depletion replay does not meet its strict trace-inventory
tolerance on the tested macOS environment. A controlled retained-state comparison
shows the same discrepancy with the original recipe. Reproducing those archived
inventories remains unresolved; their numerical records, source identities and
acceptance decisions are preserved.
