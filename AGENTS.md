# Agent instructions

Core Optimization builds OpenMC-calibrated, differentiable diffusion models for
reactor-design studies. Examples define the physical problem; shared packages
implement the reusable methods. Adding a new design may require extending the
shared packages so the example remains small.

## Code ownership

| Location | Responsibility |
| --- | --- |
| `src/geometry/` | Shared parametric build123d construction, material identities, meshing and geometry adapters |
| `src/diffusion_surrogate/` | Diffusion solvers, material calibration and response interfaces |
| `src/optimization/` | Optimizers, fitting, design loops, objectives and constraints |
| `src/depletion/` | Dense and sparse depletion integration, chain assembly and CRAM kernels |
| `src/fuel_cycle/` | Refuelling schedules, lifecycle integration and residence histories |
| `src/materials/` | Isotopic recipes, cross-section conversions and physical inventories |
| `src/coupling/` | Case interfaces and orchestration of OpenMC histories, calibration and design searches |
| `src/plotting/` | Reusable figures from explicit results and caller-defined geometry layers |
| `src/runtime/` | Serialization, workspace paths, process control and autograd memory storage |
| `src/get_nuclear_data/` | Generic nuclear-data acquisition and processing |
| `examples/<name>/` | Physical geometry, materials, scenarios, design variables, objectives and thin adapters or entry scripts |
| `tests/` | Independent numerical references, interface checks and small regression fixtures |

## Adding or extending a design

1. Inspect the relevant shared interfaces before implementing a capability. Use
   `CaseModel` and `CycleCase` to connect a coupled example to shared execution.
2. Keep examples focused on physical definitions and case-specific choices.
   Solvers, calibration engines, optimization loops and reusable data processing
   belong in `src/`.
3. When a new design reveals a missing reusable operation, extend or refactor the
   shared code as part of the task. Pass the case-specific choices through explicit
   parameters or callbacks instead of duplicating the workflow in the example.
4. Keep shared packages independent of examples: no imports from `examples`,
   example-name switches or embedded assumptions about a particular reactor.
5. Keep physical constants, geometry assumptions and genuinely case-specific
   objective definitions with the example. Extract general operations and objective
   building blocks; moving case-specific code into `src/` does not make it generic.
6. Exercise new shared behavior with a small independent case and check existing
   callers. Prefer extending existing interfaces to adding parallel frameworks or
   speculative abstractions. Preserve unrelated work.

Keep numerical methods in their domain packages. `coupling` coordinates those
methods; it does not own the depletion or fuel-movement algorithms. `runtime`
provides execution support and must not depend on case physics. Extend an existing
category before introducing another package or a catch-all utilities folder.

## Optimization goals: read, understand and clarify before proceeding

- Define new optimization goals in `examples/<name>/goals.md`. Create or update
  this file from the user's agreed goals before starting the optimization. For
  multiple scenarios, identify which goals apply to each scenario.
- Read the applicable `goals.md` before starting, resuming or changing an
  optimization. Make sure the objective, allowed design changes, constraints and
  acceptance criteria are understood before continuing.
- Record metric definitions and units, minimization or maximization directions,
  weights or priorities, parameter bounds, reference designs, relevant operating
  assumptions and stopping criteria where they affect the requested study.
- **If there is any unanswered question about the goals, stop the optimization
  work and ask the user. Always.** This includes missing requirements, ambiguous
  metrics, conflicting instructions and assumptions that could change the result.
  Do not guess, silently choose defaults or continue the dependent work while
  awaiting an answer. Apply this rule whenever uncertainty arises during a run.
- If the goals file is missing, record the user's stated goals first and ask about
  anything unresolved. Existing inputs and historical results do not establish
  agreement on new goals.
- `goals.md` records the agreed intent; the example's TOML files are its executable
  configuration. Keep them consistent. If they disagree, stop and ask the user
  before changing or running the optimization. Record clarified decisions in
  `goals.md` and update the corresponding inputs before continuing.

## Configuration and scientific contracts

- Read scenario values through the resolved example configuration. Keep one
  executable source for each setting; do not duplicate defaults in scripts or
  library code. Generated inputs are records of a run, not editable source inputs.
- Geometry follows TOML → build123d solids → shared numerical adapters.
  Keep the small geometry builder with the example. Use `CadGeometry.from_solids`
  to bind physical regions, materials and boundary faces. Extend `src/geometry/`
  for reusable operations; do not add an independent OpenMC layout, a duplicate
  FreeCAD model or a hand-written example fraction map.
- Export one `geometry.step` per prepared case and replace it when regenerating
  that case. It contains build123d solids, material identities and boundary faces.
  A saved STEP is fixed geometry. Preserve the live parametric object for shape
  derivatives; never imply that reading a STEP retains the PyTorch graph.
- Use GEOUNED in its isolated FreeCAD environment for analytic OpenMC conversion,
  and Gmsh for conformal tetrahedral meshes or the analytical sector adapter.
  Unsupported surfaces or uncertain conversion must fail without simplification
  or an automatic change of geometry backend. Verify material identities,
  boundaries, native ownership and mesh-volume convergence when changing adapters.
- For curved tetrahedral meshes, declare `volume_rtol` with the mesh inputs and
  justify it by refinement. Preparation compares mesh and physical CAD volumes;
  do not rescale physical volumes or densities to hide discretization error.
- General CAD construction and remeshing are not autograd operations. Supply an
  explicit Torch coordinate map for tetrahedral shape derivatives, or retain a
  supported exact cut-cell parameterization such as WR-1's. Keep the numerical
  geometry consistent with the CAD, and remesh when connectivity must change.
- Native and surrogate models share the physical geometry by default. A simplified
  layout requires an explicit `surrogate_mesh`, consistent material identities,
  ordering and represented volumes, and a new fit. Calibration must not silently
  choose or change the geometry. Boundary corrections are an explicit choice.
- Keep live Torch tensors connected through geometry, diffusion, depletion and
  objectives. Avoid detaching trainable quantities or converting them to Python
  numbers along a differentiated path.
- Preserve units, energy-group and isotope ordering, symmetry multiplicities,
  material volumes and inventory/energy accounting. Surface invalid contracts at
  the relevant interface instead of silently substituting another model.
- Keep surrogate predictions, OpenMC evaluations and accepted designs distinct.
  Name the selected case and state which numerical criteria it met or missed.
  Use concrete facts instead of vague status labels such as "provisional". A
  passing software test does not revalidate a retained reactor-design result.
- Preserve saved scientific evidence. Write new calculations under `runs/`; only
  figures are versioned in example result folders. Compare actual settings and
  model definitions when resuming. After changing model or solver behavior, use a
  new run and calibration. Do not weaken tolerances to make a check pass.

## Implementation and verification

- Keep validation minimal. Prefer direct operations and ordinary Python, NumPy or
  PyTorch exceptions over checks that merely rephrase missing-key, missing-file,
  type or dimension errors. Do not catch and rethrow an error just to rename it.
- Do not pin source code or configuration with hashes. Use readable input values
  and explicit model interfaces for necessary compatibility checks.
- Keep explicit checks only where normal operations could silently produce an
  incorrect result, lose gradients, reuse incompatible state or damage retained
  data. Broadcasting, truncated iteration and physical identity/order mismatches
  deserve particular attention. Validate the necessary contract once at the
  relevant boundary; avoid repeating it in every helper.
- Do not add routine shape, length or positivity guards without a concrete failure
  they prevent. Keep the numerical operation visible instead of surrounding it
  with defensive scaffolding or silent fallbacks.
- Give every function a NumPy-style docstring. Document units, array axes and
  ordering where they are part of an interface.
- Keep helpers and entry scripts small. Do not add command-line arguments unless
  requested. Apply these rules to the requested work without a wholesale rewrite.
- Use the environment described in [README.md](README.md#install). Commands below
  assume the repository's `.venv` and are run from the repository root.
- Run the smallest relevant tests for the change. For a new shared numerical
  capability, use an independent reference or invariant and check derivatives
  when applicable. Report what was checked and any skips or unresolved failures.

| Check | Command |
| --- | --- |
| Case interfaces and package independence | `.venv/bin/python -m pytest -q tests/test_case_interfaces.py tests/test_package_boundaries.py` |
| Python test suite | `.venv/bin/python -m pytest -q tests` |
| Complete bounded installation checks, including the test suite and native OpenMC | `.venv/bin/python tests/check_installation.py` |
| Minimal geometry-optimization tutorial | `.venv/bin/python -m examples.slab.run` |
| CAD, native cells and diffusion volumes for all WR-1 scenarios | `.venv/bin/python tests/check_cad_geometry.py` |

The bounded checks use small fixtures and do not require a full continuous-energy
data download. Three historical-result regressions require local WR-1 records and
may skip in a fresh checkout. Use the complete installation checks when changing
packaging or native-runtime integration; do not launch a production optimization
or large data download merely to verify a code edit.

For concurrent work, use separate case work/result directories. The current full
installation checker writes to `validation-output/`, so run it serially within a
checkout. Keep commits and remote publication within the user's requested scope.
