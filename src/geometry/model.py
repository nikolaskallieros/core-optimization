"""Named build123d solids and material/boundary assignments.

Direct build123d geometry uses millimeters. The optional analytical constructors
use centimeters and retain Torch dimensions for the existing cut-cell adapters.
"""

from dataclasses import dataclass, field
import io
import json
from pathlib import Path
import re

import build123d as bd
import torch

_STEP_RECORD = "CORE_OPTIMIZATION_GEOMETRY_V1:"


def _tensor(value):
    """Convert physical coordinates without disconnecting existing tensors.

    Parameters
    ----------
    value : array_like or torch.Tensor
        Coordinates or dimensions in cm.

    Returns
    -------
    torch.Tensor
        Float64 coordinates retaining their autograd history.
    """
    if isinstance(value, (list, tuple)):
        return torch.stack([_tensor(item) for item in value])
    return torch.as_tensor(value, dtype=torch.float64)


@dataclass(eq=False)
class Shape:
    """A build123d construction with live dimensions and shared operands.

    Parameters
    ----------
    kind : str
        Primitive or Boolean operation name.
    parameters : dict
        Named live dimensions in cm.
    children : tuple of Shape, optional
        Operands of a Boolean construction.
    """

    kind: str
    parameters: dict = field(default_factory=dict)
    children: tuple = ()

    def __sub__(self, other):
        """Subtract a solid using the same operation in every adapter.

        Parameters
        ----------
        other : Shape
            Solid to remove.

        Returns
        -------
        Shape
            Boolean difference.
        """
        return Shape("difference", children=(self, other))

    def __and__(self, other):
        """Intersect two solids without copying their physical definitions.

        Parameters
        ----------
        other : Shape
            Clipping solid.

        Returns
        -------
        Shape
            Boolean intersection.
        """
        return Shape("intersection", children=(self, other))

    def slice(self, z_bounds_cm):
        """Restrict a construction to an axial interval.

        Parameters
        ----------
        z_bounds_cm : array_like or torch.Tensor
            Bottom and top elevations in cm.

        Returns
        -------
        Shape
            Shared axial slice, retaining the original transverse construction.
        """
        return Shape("slice", {"z": _tensor(z_bounds_cm)}, (self,))

    def build(self, cache=None):
        """Evaluate the construction with build123d in millimeters.

        Parameters
        ----------
        cache : dict or None, optional
            Per-evaluation cache for shared operands. Never retain it after changing
            live dimensions.

        Returns
        -------
        build123d.Shape
            Boundary representation of the current physical solid.
        """
        cache = {} if cache is None else cache
        if self in cache:
            return cache[self]
        values = {name: value.detach().cpu().numpy() * 10
                  for name, value in self.parameters.items()}
        if self.kind == "box":
            lower, upper = values["lower"], values["upper"]
            result = bd.Solid.make_box(*(upper - lower), bd.Plane(origin=tuple(lower)))
        elif self.kind == "cylinder":
            xy, radius, bounds = values["xy"], values["radius"], values["z"]
            result = bd.Solid.make_cylinder(float(radius), float(bounds[1] - bounds[0]),
                                           bd.Plane(origin=(*xy, bounds[0])))
        elif self.kind == "prism":
            xy, bounds = values["vertices"], values["z"]
            wire = bd.Wire.make_polygon([(*point, bounds[0]) for point in xy], close=True)
            result = bd.Solid.extrude(bd.Face(wire), (0., 0., float(bounds[1] - bounds[0])))
        elif self.kind == "slice":
            lower, upper = (value.detach().cpu().numpy() * 10 for value in self.children[0].bounds())
            lower[2], upper[2] = values["z"]
            slab = bd.Solid.make_box(*(upper - lower), bd.Plane(origin=tuple(lower)))
            result = self.children[0].build(cache).intersect(slab)
        else:
            operands = [child.build(cache) for child in self.children]
            if self.kind == "union":
                result = operands[0].fuse(*operands[1:]) if len(operands) > 1 else operands[0]
            elif self.kind == "difference":
                result = operands[0].cut(operands[1])
            elif self.kind == "intersection":
                result = operands[0].intersect(operands[1])
            else:
                raise ValueError(f"Unknown CAD construction: {self.kind}")
        if isinstance(result, bd.ShapeList):
            result = bd.Compound(result)
        cache[self] = result
        return result

    def bounds(self):
        """Return a live Cartesian enclosure of the solid.

        Returns
        -------
        tuple of torch.Tensor
            Lower and upper x/y/z coordinates in cm. Boolean differences retain
            the first operand's enclosure, which may include removed space.
        """
        values = self.parameters
        if self.kind == "box":
            return values["lower"], values["upper"]
        if self.kind == "cylinder":
            return (torch.cat((values["xy"] - values["radius"], values["z"][:1])),
                    torch.cat((values["xy"] + values["radius"], values["z"][1:])))
        if self.kind == "prism":
            return (torch.cat((values["vertices"].amin(0), values["z"][:1])),
                    torch.cat((values["vertices"].amax(0), values["z"][1:])))
        bounds = [child.bounds() for child in self.children]
        if self.kind == "slice":
            lower, upper = bounds[0]
            return (torch.cat((lower[:2], torch.maximum(lower[2:3], values["z"][:1]))),
                    torch.cat((upper[:2], torch.minimum(upper[2:3], values["z"][1:]))))
        if self.kind == "difference":
            return bounds[0]
        lower = torch.stack([pair[0] for pair in bounds])
        upper = torch.stack([pair[1] for pair in bounds])
        return ((lower.amin(0), upper.amax(0)) if self.kind == "union"
                else (lower.amax(0), upper.amin(0)))


def box(lower_cm, upper_cm):
    """Define an axis-aligned build123d box from two opposite corners.

    Parameters
    ----------
    lower_cm, upper_cm : array_like or torch.Tensor
        Lower and upper x/y/z coordinates in cm.

    Returns
    -------
    Shape
        Parametric box shared by CAD and numerical adapters.
    """
    return Shape("box", {"lower": _tensor(lower_cm), "upper": _tensor(upper_cm)})


def cylinder(xy_cm, radius_cm, z_bounds_cm):
    """Define a build123d cylinder parallel to the physical z axis.

    Parameters
    ----------
    xy_cm : array_like or torch.Tensor
        Cylinder center in cm.
    radius_cm : float or torch.Tensor
        Outer radius in cm.
    z_bounds_cm : array_like or torch.Tensor
        Bottom and top elevations in cm.

    Returns
    -------
    Shape
        Parametric cylinder shared by CAD and numerical adapters.
    """
    return Shape("cylinder", {"xy": _tensor(xy_cm), "radius": _tensor(radius_cm),
                              "z": _tensor(z_bounds_cm)})


def prism(vertices_cm, z_bounds_cm):
    """Extrude a counterclockwise convex polygon with build123d.

    Parameters
    ----------
    vertices_cm : array_like or torch.Tensor
        Counterclockwise transverse vertices, shaped (n_vertices, 2), in cm.
    z_bounds_cm : array_like or torch.Tensor
        Bottom and top elevations in cm.

    Returns
    -------
    Shape
        Parametric polygonal prism. Convexity supports exact material cuts in
        the analytical diffusion adapter.
    """
    vertices = _tensor(vertices_cm)
    edges = torch.roll(vertices, -1, 0) - vertices
    following = torch.roll(edges, -1, 0)
    turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
    if bool((turns.detach() <= 0).any()):
        raise ValueError("CAD prisms require a convex counterclockwise polygon")
    return Shape("prism", {"vertices": vertices, "z": _tensor(z_bounds_cm)})


def union(shapes):
    """Join material pieces into one named region.

    Parameters
    ----------
    shapes : iterable of Shape
        Physical pieces belonging to the same material region.

    Returns
    -------
    Shape
        Shared Boolean union.
    """
    return Shape("union", children=tuple(shapes))


@dataclass
class MaterialVolume:
    """A physical volume with one stable identity and material assignment.

    Parameters
    ----------
    label : str
        Stable region identity, also used for native tallies and diffusion columns.
    material : str
        Key into the caller's material definitions; not a material composition.
    shape : build123d.Shape or Shape
        Actual CAD solids in mm, or an analytical construction with live dimensions.
    """

    label: str
    material: str
    shape: bd.Shape | Shape


@dataclass
class CadGeometry:
    """One physical CAD model consumed by both solver adapters.

    Parameters
    ----------
    domain : Shape or None
        Analytical outer domain, or None when the build123d volumes define it.
    volumes : tuple of MaterialVolume
        Ordered nonoverlapping material volumes that partition the domain.
    boundaries : dict
        Boundary names mapped to specifications with ``type`` and ``faces``.
        Faces are actual build123d faces in mm. Unassigned exterior faces use
        vacuum conditions. Analytical adapters also accept their named faces.
    name : str, optional
        Human-readable CAD assembly name.

    Notes
    -----
    Adapters consume this object or its STEP export. They never read an example's
    TOML or invent a second physical geometry. Live tensors are preserved when
    consuming the object; reading a saved STEP restores a fixed geometry.
    """

    domain: Shape | None
    volumes: tuple[MaterialVolume, ...]
    boundaries: dict
    name: str = "geometry"

    @classmethod
    def from_solids(cls, solids, *, materials=None, boundaries=None, name="geometry"):
        """Bind material identities to direct build123d solids.

        Parameters
        ----------
        solids : mapping of str to build123d.Shape
            Region labels and their nonoverlapping solids in mm. Boolean results,
            curved solids and disconnected compounds are accepted unchanged.
        materials : mapping of str to str or None, optional
            Region-to-material keys; None uses each region label as its key.
        boundaries : dict or None, optional
            Named face selections and their boundary conditions.
        name : str, optional
            Assembly name.

        Returns
        -------
        CadGeometry
            Geometry without a package-specific primitive representation.
        """
        materials = materials if materials is not None else {label: label for label in solids}
        return cls(None, tuple(MaterialVolume(label, materials[label], solid)
                               for label, solid in solids.items()), boundaries or {}, name)

    @property
    def is_direct(self):
        """Identify geometry supplied as actual CAD solids.

        Returns
        -------
        bool
            True for the direct build123d interface.
        """
        return self.domain is None

    def bounds_cm(self):
        """Return the physical enclosing box in solver units.

        Returns
        -------
        tuple of torch.Tensor
            Lower and upper coordinates in cm.
        """
        if not self.is_direct:
            return self.domain.bounds()
        bounds = self.build().bounding_box()
        return _tensor(tuple(bounds.min)) / 10, _tensor(tuple(bounds.max)) / 10

    def as_solids(self):
        """Build an analytical parameterization into the direct CAD interface.

        Returns
        -------
        CadGeometry
            Actual build123d solids and boundary-face selections in mm. Native
            adapters consume these solids; live parameters remain in this
            original object for the optional exact cut-cell diffusion mapping.
        """
        if self.is_direct:
            return self
        assembly = self.build()
        faces = tuple(self.domain.build().faces())
        targets = {"bottom": (0., 0., -1.), "top": (0., 0., 1.)}
        if self.domain.kind == "box":
            targets.update(x_min=(-1., 0., 0.), x_max=(1., 0., 0.),
                           y_min=(0., -1., 0.), y_max=(0., 1., 0.))
        elif self.domain.kind == "prism":
            vertices = self.domain.parameters["vertices"].detach()
            for i, (a, b) in enumerate(zip(vertices, vertices.roll(-1, 0))):
                delta = b - a
                normal = torch.stack((delta[1], -delta[0], delta.new_zeros(())))
                targets[f"side/{i}"] = tuple(normal / normal.norm())
        selections = {}
        for name, specification in self.boundaries.items():
            selected = [face for face in faces if
                        (face.geom_type == bd.GeomType.CYLINDER if name == "side" else
                         face.geom_type == bd.GeomType.PLANE and
                         bd.Vector(targets[name]).dot(face.normal_at()) > 1 - 1e-10)]
            selections[name] = {**specification, "faces": tuple(selected)}
        return CadGeometry.from_solids({child.label: child for child in assembly.children},
                                       materials={v.label: v.material for v in self.volumes},
                                       boundaries=selections, name=self.name)

    def record(self):
        """Describe the current CAD construction and material bindings.

        Returns
        -------
        dict
            Analytical constructions include shared operands and current dimensions
            in cm. Direct CAD records contain only material/boundary metadata;
            their physical solids must be carried by the STEP file.
        """
        if self.is_direct:
            return {"format": "build123d", "name": self.name, "length_unit": "mm",
                    "volumes": [{"label": v.label, "material": v.material} for v in self.volumes],
                    "boundaries": {name: {key: value for key, value in spec.items() if key != "faces"}
                                   for name, spec in self.boundaries.items()}}
        nodes, indices = [], {}

        def visit(shape):
            """Store one shared construction node.

            Parameters
            ----------
            shape : Shape
                Construction to encode.

            Returns
            -------
            int
                Node position in dependency order.
            """
            if shape not in indices:
                children = [visit(child) for child in shape.children]
                indices[shape] = len(nodes)
                nodes.append({"kind": shape.kind, "children": children,
                              "parameters": {key: value.detach().cpu().tolist()
                                             for key, value in shape.parameters.items()}})
            return indices[shape]

        domain = visit(self.domain)
        volumes = [{"label": volume.label, "material": volume.material,
                    "shape": visit(volume.shape)} for volume in self.volumes]
        return {"name": self.name, "length_unit": "cm", "nodes": nodes,
                "domain": domain, "volumes": volumes, "boundaries": self.boundaries}

    @classmethod
    def from_record(cls, record):
        """Restore a fixed physical geometry from its generated construction record.

        Parameters
        ----------
        record : dict
            Record produced by ``record``; coordinates must be in cm.

        Returns
        -------
        CadGeometry
            Restored shared model with constant Torch dimensions.
        """
        if record["length_unit"] != "cm":
            raise ValueError("CAD construction coordinates must be in cm")
        nodes = []
        for node in record["nodes"]:
            nodes.append(Shape(node["kind"], {key: _tensor(value) for key, value in node["parameters"].items()},
                               tuple(nodes[index] for index in node["children"])))
        return cls(nodes[record["domain"]], tuple(MaterialVolume(volume["label"], volume["material"],
                                                               nodes[volume["shape"]])
                                                  for volume in record["volumes"]),
                   record["boundaries"], record["name"])

    def build(self):
        """Build the named material-volume assembly with build123d.

        Returns
        -------
        build123d.Compound
            CAD assembly in mm, retaining one child per material region.
        """
        cache, children = {}, []
        for volume in self.volumes:
            shape = volume.shape if self.is_direct else (self.domain & volume.shape).build(cache)
            child = bd.Compound(list(shape.solids()), label=volume.label)
            children.append(child)
        return bd.Compound(label=self.name, children=children)

    def write_step(self, path):
        """Replace one self-contained CAD file with the current design.

        Parameters
        ----------
        path : str or pathlib.Path
            STEP destination. Material and boundary bindings are embedded in
            the standard STEP description; no sidecar geometry file is written.

        Returns
        -------
        pathlib.Path
            Written STEP file, containing actual build123d solids in mm.
        """
        if not self.is_direct:
            return self.as_solids().write_step(path)
        path = Path(path)
        stream = io.BytesIO()
        assembly = self.build()
        assembly = bd.Compound(label=self.name, children=[*assembly.children, *(
            bd.Compound(list(spec["faces"]), label=f"boundary/{name}")
            for name, spec in self.boundaries.items())])
        bd.export_step(assembly, stream, unit=bd.Unit.MM)
        text = stream.getvalue().decode()
        record = json.dumps(self.record(), separators=(",", ":"))
        # OpenCascade's STEP scanner has a finite string buffer. Small standard
        # description entries keep large reactor constructions interoperable.
        chunks = [record[start:start + 2048] for start in range(0, len(record), 2048)]
        descriptions = [f"'{_STEP_RECORD}{index}:" + chunk.replace("'", "''") + "'"
                        for index, chunk in enumerate(chunks)]
        text = text.replace("FILE_DESCRIPTION((", "FILE_DESCRIPTION((" + ",\n".join(descriptions) + ",\n", 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)
        return path

    @classmethod
    def read_step(cls, path):
        """Read the shared physical model and materials from its STEP artifact.

        Parameters
        ----------
        path : str or pathlib.Path
            STEP file containing actual CAD solids. Package exports also carry
            material identities and boundary faces; ordinary STEP files use the
            assembly labels as material keys and exterior vacuum boundaries.

        Returns
        -------
        CadGeometry
            Fixed model. Use the original parametric object for shape gradients;
            a STEP snapshot cannot serialize a live PyTorch computation graph.
        """
        text = Path(path).read_text()
        chunks = re.findall("'" + _STEP_RECORD + r"(\d+):((?:''|[^'])*)'", text)
        if not chunks:
            assembly = bd.import_step(path)
            children = assembly.children or tuple(assembly.solids())
            labels = [child.label or f"region_{i + 1}" for i, child in enumerate(children)]
            if len(set(labels)) != len(labels):
                raise ValueError("STEP regions must have distinct labels for material assignment")
            return cls.from_solids(dict(zip(labels, children)), name=assembly.label or "geometry")
        record = "".join(value.replace("''", "'") for _, value in sorted(chunks, key=lambda item: int(item[0])))
        record = json.loads(record)
        children = {child.label: child for child in bd.import_step(path).children}
        boundaries = {name: {**spec, "faces": tuple(children[f"boundary/{name}"].faces())}
                      for name, spec in record["boundaries"].items()}
        return cls.from_solids({v["label"]: children[v["label"]] for v in record["volumes"]},
                               materials={v["label"]: v["material"] for v in record["volumes"]},
                               boundaries=boundaries, name=record["name"])
