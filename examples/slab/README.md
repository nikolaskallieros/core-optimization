# A small geometry optimization

This tutorial varies the width of a homogeneous slab until its effective
multiplication factor reaches a target. It shows the complete path from a
readable input file to geometry, a diffusion solution, a derivative and an
optimized dimension. All numerical algorithms come from the shared packages.

After [installing the project](../../README.md#install), run from the repository root:

```sh
.venv/bin/python -m examples.slab.run
```

The default case produces approximately:

```text
Width: 20.000 → 8.631 cm; keff: 1.509746 → 1.000000
Initial dkeff/dwidth: 0.024150 /cm
```

The script also prints the measured optimization time. The model uses
manufactured one-group cross sections; it needs no OpenMC run or nuclear-data
download and is not a calibrated reactor model. It writes the final build123d CAD
to `runs/slab/geometry.step`, replacing that file on the next run.

## Follow the information

```text
model.toml → Slab.geometry() → shared CAD → diffusion → objective
                                  ↓                       ↓
                             geometry.step       gradients and updated width
```

| File or interface | What you supply or receive |
| --- | --- |
| [model.toml](model.toml) | Cell count, initial width, bounds, material data and target k-effective |
| [Slab](run.py) | One trainable coordinate, fixed material data and a geometry builder |
| `CadGeometry` | One build123d construction, named material volumes and face boundary conditions |
| `geometry.diffusion.discretize` | Mesh and exact material fractions derived from that CAD construction |
| `geometry.openmc.from_cad` | OpenMC geometry derived from the same construction or its saved STEP |
| `Diffusion3D.solve` | Cell-by-region fractions and macroscopic data in; k-effective, flux and power out |
| `torch.autograd.grad` | The initial sensitivity of k-effective to width |
| `minimize` | A scalar objective and a PyTorch optimizer; updated parameters and a convergence record |

The slab has vacuum boundaries at its two ends and reflective transverse faces.
One cell in each transverse direction makes this a one-dimensional problem on
the three-dimensional solver interface. Each cell contains the same material,
so its single material fraction is one.

`Slab.forward()` constructs the CAD model from the **live width tensor** on
every evaluation. The shared adapter derives the mesh edges and material fractions.
Converting that width to a Python float or detaching it along the differentiated
path would sever the geometry derivative. A sigmoid maps the
optimizer's unconstrained coordinate into the physical width bounds, keeping
line-search trials valid. The physical sensitivity printed by the script is
`dkeff/dwidth`, not the derivative with respect to that internal coordinate.

The objective is `(keff - target_keff)**2`. L-BFGS evaluates the model and its
first derivatives through the shared `minimize` interface. The fixed material
uses isotropic, nonmultiplying scattering and `D = 1 / (3 * transport)`.

For the default material, the continuum slab solution gives a critical width
of approximately **8.637 cm** with the solver's vacuum boundary condition.
The 24-cell result differs slightly because of spatial discretization. The
[tutorial regression](../../tests/test_slab_tutorial.py) checks this independent
analytical solution as well as the optimized target and sensitivity sign.

## Make a new example

Copy this directory to `examples/my_case`, then edit `model.toml` and the geometry
in `run.py`. Run it with `.venv/bin/python -m examples.my_case.run`.

For a new 3D layout, build the actual solids with build123d and bind their material
keys with `CadGeometry.from_solids`. The adapters derive analytic OpenMC cells
through GEOUNED and a conformal tetrahedral diffusion mesh through Gmsh.
The slab's analytical `box` construction retains its exact width derivative;
that optional parameterization is useful for simple moving dimensions.
Supply cross sections in the same material-volume order; scattering arrays use
incoming then outgoing energy-group axes. A different objective only needs to
return a scalar Torch tensor from the solver responses.

For example, the existing slab geometry can also produce OpenMC cells:

```python
from geometry.openmc import from_cad

cad = model.geometry()
cad.write_step("runs/slab/geometry.step")
native = from_cad(cad, {"fuel": openmc_material})
```

`openmc_material` is the caller's OpenMC material for the TOML material key.
This constructs geometry without running transport. Passing the STEP path to
either adapter restores a fixed model, including material identities. Keep the
live parameterization for differentiation: STEP stores fixed solids, not
PyTorch's computation graph. Analytical helper lengths are in cm; direct
build123d and STEP lengths are in mm.

The box mesh is Cartesian. The WR-1 example uses Gmsh to mesh the CAD sector face
and exact cylinder intersections to retain moving-interface derivatives.
General solids use `discretize(cad, max_size_cm=...)` and
`FiniteElementDiffusion(mesh.grid)`. Material derivatives work directly; shape
derivatives require an explicit Torch mesh-coordinate map consistent with the CAD.
The tetrahedral mesh is fixed between remeshing steps. Current exact analytical
cylinder intersections support disjoint or nested cylinders, including annuli.

For an OpenMC-calibrated depletion study, use the
[WR-1 case adapter](../wr_1/model/case.py) as the next example. It binds geometry,
a fresh isotope inventory, surrogate construction and fitting targets through
[`CaseModel`](../../src/coupling/cases.py). `CycleCase` adds resolved
scenario inputs and explicit output paths. The shared `prepare_case` and
`run_case` functions handle native preparation, histories, calibration and the
optimization loop; those engines belong in `src/` and do not need copying.

`CaseModel.mesh(design, native_case)` supplies a `DiffusionGeometry` built from
the native physical definition. Leave `surrogate_mesh` unset to use that shared
geometry. Set it explicitly for a simplified material layout, preserving region
labels and volumes. The calibration factory receives the selected mapping through
its `mesh` argument, so it does not independently choose a geometry. Cross-section
and diffusion corrections use the generic `CalibratedDiffusion` interface; optional
spatial comparisons are supplied separately from the calibration equations.
