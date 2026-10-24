"""Explicit adapters and locations for a native/surrogate optimization case."""

from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
import tomllib
from typing import Callable, Mapping


def _merge(defaults, overrides):
    """Merge scenario overrides into an independent copy of shared defaults.

    Parameters
    ----------
    defaults : dict
        Base configuration mapping.
    overrides : dict
        Scenario-specific values; lists, scalars and reference objects replace their
        defaults.

    Returns
    -------
    dict
        Merged configuration with recursively combined ordinary mappings.
    """
    result = deepcopy(defaults)
    for key, value in overrides.items():
        result[key] = (_merge(result[key], value) if isinstance(result.get(key), dict)
                       and isinstance(value, dict) and "$ref" not in result[key]
                       and "$ref" not in value else deepcopy(value))
    return result


def _resolve(value, document, references=()):
    """Resolve internal configuration references without mutating their source.

    Parameters
    ----------
    value : object
        Configuration subtree, scalar or reference object to resolve.
    document : dict
        Assembled settings and data document used as the reference root.
    references : tuple of str, optional
        References currently being expanded, used to detect cycles.

    Returns
    -------
    object
        Resolved subtree with reference objects replaced by their target values.

    Raises
    ------
    ValueError
        A reference is malformed, cyclic or mixed with other fields.
    """
    if isinstance(value, dict):
        if "$ref" in value and len(value) != 1:
            raise ValueError("A configuration reference cannot contain other fields")
        if set(value) == {"$ref"}:
            reference = value["$ref"]
            if not reference.startswith("#/"):
                raise ValueError("Configuration references must start with #/")
            if reference in references:
                raise ValueError(f"Cyclic configuration reference: {reference}")
            target = document
            for key in reference[2:].split("/"):
                key = key.replace("~1", "/").replace("~0", "~")
                target = target[int(key)] if isinstance(target, list) else target[key]
            return _resolve(target, document, (*references, reference))
        return {key: _resolve(item, document, references) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, document, references) for item in value]
    return value


@dataclass(frozen=True)
class Configuration:
    """One resolved scenario and its input source paths.

    Accessors return copies so callers cannot modify the authoritative snapshot.
    Changed input values require loading a new configuration before execution.
    """
    path: Path
    scenario: str
    files: Mapping[str, Path]
    _document: Mapping

    def settings(self, kind):
        """Return an independent copy of a resolved settings section.

        Parameters
        ----------
        kind : str
            Configuration section name, such as cycle, calibration or optimization.

        Returns
        -------
        dict
            Effective settings after merging defaults, scenario overrides and references.
        """
        return deepcopy(self._document["settings"][kind])

    def data(self, name):
        """Return an independent copy of a declared input definition.

        Parameters
        ----------
        name : str
            Name declared in the configuration's data mapping.

        Returns
        -------
        dict
            Resolved model or nuclear-data source definition.
        """
        return deepcopy(self._document["data"][name])

    def verify_sources(self):
        """Reject changed input values after the configuration was loaded.

        Raises
        ------
        ValueError
            Effective input values differ from the loaded snapshot.
        """
        if load_configuration(self.path, self.scenario).resolved() != self.resolved():
            raise ValueError("Configuration input changed after loading; reload it")

    def resolved(self):
        """Return the complete effective input snapshot for inspection.

        Returns
        -------
        dict
            Independent copy of the scenario identity, resolved settings and declared data.
        """
        return {"scenario": self.scenario, **deepcopy(self._document)}


def load_configuration(path, scenario):
    """Load and resolve shared defaults, scenario overrides and references.

    Parameters
    ----------
    path : str or pathlib.Path
        Study TOML or JSON file declaring schema version 1, scenarios and optional relative
        data paths.
    scenario : str or int
        Scenario key to select after conversion to a string.

    Returns
    -------
    Configuration
        Resolved input values and their source paths.

    Raises
    ------
    ValueError
        The schema, file type, data name or reference graph is invalid.

    Notes
    -----
    Dictionaries merge recursively; lists and scalars replace earlier values. A mapping
    containing only ``$ref`` resolves a path such as
    ``#/settings/optimization/bounds/cycle_days`` in the assembled document. Missing or
    cyclic references fail before model construction. No calculation defaults are invented.
    """
    path = Path(path).resolve()
    files = {"configuration": path}

    def read(source):
        """Parse one declared configuration source.

        Parameters
        ----------
        source : pathlib.Path
            TOML or JSON source file.

        Returns
        -------
        dict
            Decoded source document.
        """
        raw = source.read_bytes()
        if source.suffix == ".toml":
            return tomllib.loads(raw.decode("utf-8"))
        if source.suffix == ".json":
            return json.loads(raw)
        raise ValueError(f"Configuration inputs must be TOML or JSON: {source}")

    record = read(path)
    if record.get("schema_version") != 1:
        raise ValueError("Unsupported case configuration schema")
    scenario = str(scenario)
    overrides = record["scenarios"][scenario]
    data = {}
    for name, relative in record.get("data", {}).items():
        if name == "configuration":
            raise ValueError("The data name 'configuration' is reserved")
        files[name] = (path.parent / relative).resolve()
        data[name] = read(files[name])
    document = {"data": data, "settings": _merge(record.get("defaults", {}), overrides)}
    return Configuration(path, scenario, files, _resolve(document, document))


@dataclass(frozen=True)
class CaseModel:
    """Case physics supplied to the shared cycle, calibration and search engines.

    Factories receive physical values, not a repository root or an example name.
    design() returns a Torch module with the declared trainable parameters.
    transport(**values) returns the native region/material description;
    inventory(values) supplies its isotope inventories and fresh-fuel recipe.
    surrogate(design, native_state, layout, mesh=...) returns the calibrated diffusion
    model. equivalence_bases(native_case, **options) selects fitting groups;
    fitting_targets(native_result, native_case) supplies weighted observations.

    Optional export_transport(native_case, directory) writes native inputs;
    mesh(design, native_case) discretizes the native geometry as DiffusionGeometry.
    surrogate_mesh(design, native_case) explicitly replaces that default mapping when
    a simplified surrogate geometry is wanted. Both preserve native region identities
    and material volumes; changed region sets require an explicit homogenization and
    response mapping, which this interface does not infer.
    initialize_loading(values, keff, options) proposes the next loading.
    """

    design: Callable
    inventory: Callable
    transport: Callable
    surrogate: Callable
    equivalence_bases: Callable
    fitting_targets: Callable
    initialize_loading: Callable | None = None
    export_transport: Callable | None = None
    mesh: Callable | None = None
    surrogate_mesh: Callable | None = None

    def make_mesh(self, design, native_case):
        """Discretize native geometry unless an alternative is explicitly selected.

        Parameters
        ----------
        design : torch.nn.Module
            Caller-owned live design supplying physical dimensions.
        native_case : mapping
            Native physical region and geometry definition. An alternative mapper receives
            an isolated copy so it cannot modify the native model.

        Returns
        -------
        DiffusionGeometry or None
            Selected mesh and live material mapping. None applies to models that declare
            no spatial discretization.

        Raises
        ------
        ValueError
            The mapper changes physical region identities or their order without a declared
            homogenization and response mapping.
        """
        factory = self.mesh if self.surrogate_mesh is None else self.surrogate_mesh
        if factory is None:
            return None
        definition = native_case if self.surrogate_mesh is None else deepcopy(native_case)
        geometry = factory(design, definition)
        if tuple(geometry.region_labels) != tuple(region["label"] for region in native_case["regions"]):
            raise ValueError("Surrogate geometry must preserve native region identities and order; "
                             "merging regions requires an explicit homogenization/response mapping")
        return geometry

    def make_surrogate(self, design, native_state, layout):
        """Build a calibrated model with the same geometry policy used in preparation.

        Parameters
        ----------
        design : torch.nn.Module
            Caller-owned live design passed to geometry and material factories.
        native_state : mapping
            Native case and measured responses; its case selects the physical region layout.
        layout : str
            Caller-defined calibration-basis grouping.

        Returns
        -------
        torch.nn.Module
            Calibrated model constructed from the selected DiffusionGeometry.
        """
        return self.surrogate(design, native_state, layout,
                              mesh=self.make_mesh(design, native_state["case"]))


@dataclass(frozen=True)
class CycleCase:
    """Bind a resolved configuration, model factories and explicit output paths.

    Independent adapters may instead supply three section files through inputs.
    A resolved configuration owns those paths when one is supplied.
    """

    name: str
    results: Path
    work: Path
    model: CaseModel
    archive_root: Path
    inputs: Mapping[str, Path] | None = None
    initialize_loading: bool = True
    result_prefix: str = ""
    data_subdirectory: str = "data"
    configuration: Configuration | None = None

    def __post_init__(self):
        """Normalize artifact paths and resolve the declared input sources.

        Raises
        ------
        ValueError
            Input paths conflict with the resolved configuration.
        """
        for name in ("results", "work", "archive_root"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        inputs = {key: Path(value) for key, value in (self.inputs or {}).items()}
        if self.configuration is not None:
            configured = {kind: self.configuration.path for kind in ("cycle", "calibration", "optimization")}
            if inputs and inputs != configured:
                raise ValueError("Input paths conflict with the resolved configuration")
            inputs = configured
        object.__setattr__(self, "inputs", inputs)

    def input(self, kind="cycle"):
        """Return the source path associated with a settings section.

        Parameters
        ----------
        kind : str, optional
            Configuration section name, such as cycle, calibration or optimization.

        Returns
        -------
        pathlib.Path
            Source file path; use ``settings`` to obtain effective values.
        """
        return self.inputs[kind]

    @property
    def data(self):
        """Return the directory for numerical result records.

        Returns
        -------
        pathlib.Path
            Data subdirectory within the case's result directory.
        """
        return self.results / self.data_subdirectory

    @property
    def figures(self):
        """Return the directory for generated result figures.

        Returns
        -------
        pathlib.Path
            Figures subdirectory within the case's result directory.
        """
        return self.results / "figures"

    def result(self, kind, suffix=".json"):
        """Resolve the artifact path for a named result.

        Parameters
        ----------
        kind : str
            Result identity; status is stored in the working directory.
        suffix : str, optional
            Filename extension; an empty suffix selects a result directory and image extensions
            select figures.

        Returns
        -------
        pathlib.Path
            Path with the case prefix and appropriate artifact directory.
        """
        if kind == "status":
            return self.work / "status.json"
        directory = self.results if not suffix else self.data
        if suffix in (".png", ".svg", ".pdf"):
            directory = self.figures
        return directory / f"{self.result_prefix}{kind}{suffix}"

    def prepare_outputs(self):
        """Create the numerical-result, figure and working directories."""
        for directory in (self.data, self.figures, self.work):
            directory.mkdir(parents=True, exist_ok=True)

    def native_states(self):
        """Resolve the retained native-state directory within the case archive.

        Returns
        -------
        pathlib.Path
            Local state directory or its manifest-declared source directory.

        Raises
        ------
        ValueError
            The manifest points outside the declared archive root.
        """
        directory = self.result("states", "")
        manifest = json.loads((directory / "manifest.json").read_text())
        if "source_directory" in manifest:
            source = (directory / manifest["source_directory"]).resolve()
            if not source.is_relative_to(self.archive_root.resolve()):
                raise ValueError("Native baseline reference must remain inside the case archive")
            return source
        return directory

    def settings(self, kind="cycle"):
        """Return authoritative settings for the requested calculation stage.

        Parameters
        ----------
        kind : str, optional
            Configuration section name, such as cycle, calibration or optimization.

        Returns
        -------
        dict
            Resolved configuration values, or the contents of a legacy section JSON file.
        """
        return (json.loads(self.input(kind).read_text()) if self.configuration is None else
                self.configuration.settings(kind))

    def design(self):
        """Construct the live design through the declared model factory.

        Returns
        -------
        torch.nn.Module
            Design module exposing the case's named trainable parameters.
        """
        return self.model.design()

    def inventory(self, values=None):
        """Construct the physical inventory for the requested design.

        Parameters
        ----------
        values : dict or None, optional
            Named design values passed to the inventory factory; None requests its initial
            design.

        Returns
        -------
        object
            Physical inventory object supplied by the case's inventory factory.
        """
        return self.model.inventory(values)

    def make_case(self, **kwargs):
        """Construct the native case through the declared transport factory.

        Parameters
        ----------
        **kwargs : dict
            Transport-factory inputs, such as design_values, fuel_states, boron or sampling.

        Returns
        -------
        dict
            Native geometry, material and transport settings.
        """
        return self.model.transport(**kwargs)

    def definition(self):
        """Return effective inputs and geometry policy for restart comparison.

        Returns
        -------
        dict
            Readable case name, scenario, resolved settings, input definitions and
            native or explicitly simplified surrogate geometry policy.
        """
        inputs = (self.configuration.resolved() if self.configuration is not None else
                  {"settings": {kind: self.settings(kind) for kind in self.inputs}})
        return {"case_name": self.name, **inputs,
                "surrogate_geometry": "native" if self.model.surrogate_mesh is None else "explicit"}
