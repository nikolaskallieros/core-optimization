# Third-party material

The depletion coefficient tables and chain/CRAM implementation in
`src/depletion/kernels/` derive from
[OpenMC](https://github.com/openmc-dev/openmc). Their original
[MIT license notice](src/depletion/kernels/LICENSE) is included
beside the code.

OpenMC, PyTorch, [build123d](https://github.com/gumyr/build123d),
[Gmsh](https://gmsh.info/) and the other installed dependencies retain their own licenses.
They are installed separately.

The analytic CAD adapter invokes [GEOUNED](https://github.com/GEOUNED-org/GEOUNED)
(EUPL-1.2) and [FreeCAD](https://github.com/FreeCAD/FreeCAD/blob/1.0.0/LICENSE) (LGPL-2.1-or-later)
in a separate environment. Their source is not vendored in this repository.

The reports include public-source research, excerpts and figures with source
attribution. Availability on the web does not change the source material's
copyright. A project-wide license has not yet been selected.
