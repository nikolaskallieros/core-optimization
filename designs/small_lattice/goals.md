# Small lattice workflow

Run a self-contained local demonstration of the complete CAD → OpenMC reference
history → diffusion calibration → differentiable depletion and optimization →
new OpenMC history workflow. The user agreed to target k-effective = 1.0 by
changing one common fuel radius and to include a short depletion history.

Use the nine-pin square lattice and manufactured one-group data in `inputs/model.toml`.
The editable dimensions, cross sections, chain values, sampling and numerical
criteria have one executable source in that file. These are illustrative inputs,
not an evaluated-data reactor design. The model has fuel cylinders and surrounding
coolant; cladding, thermal feedback and control systems are omitted. Reflective
outer faces represent a repeated lattice with no leakage.

Minimize the squared difference between **end-of-history k-effective and 1.0**.
Only the common fuel radius changes, within the bounds in `inputs/model.toml`. Pin count,
pitch, height, initial isotope densities, prescribed total power and history
durations stay fixed. Every candidate starts from the same fresh number densities;
its physical fuel mass therefore changes with its radius. There is no refuelling.
The prescribed-power depletion histories report their actual eigenvalues; they
do not claim controlled critical operation throughout the history.

Fit one positive material diffusion correction to each native reference's final
k-effective. Keep it fixed during the following design search. Acceptance of the
final design requires a new OpenMC history with final |k-effective − 1| no greater
than max(0.005, 3 reported standard errors), and standard error at most 0.002.
The reported standard error is conditional on that transport state's inventory;
it does not propagate Monte Carlo uncertainty through earlier depletion stages.
Allow two design proposals, retaining an improving or target-satisfying candidate.

Verify CAD/material-volume agreement, native geometry ownership, power/energy
accounting, a depletion step against an independent matrix exponential, the full
history's radius derivative against centered differences, and the final surrogate
on a finer mesh without refitting. Preserve the native inputs, logs, isotope
histories, optimization records and checks under a fresh `runs/small_lattice/`
execution directory. Keep a concise explanation and reproducible command here.
