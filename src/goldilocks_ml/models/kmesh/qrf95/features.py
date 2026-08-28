"""The verified 483-column QRF95 feature contract.

Adapted from ``stfc/goldilocks_kpoints``
(``utils/compound_features_utils.py``), © 2024 Science and Technology
Facilities Council. Redistributed under this repository's LICENSE; the
upstream licence is under review in stfc/goldilocks-ml#10.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from dscribe.descriptors import SOAP
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
)
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from goldilocks_ml.models.metallicity.cgcnn import (
    REPRESENTATION_WIDTH,
    crystal_representations,
)
from goldilocks_ml.protocol import TrainingProtocol
from goldilocks_ml.registry import FeatureMatrix, register_feature_contract
from goldilocks_ml.snapshot import Snapshot

SCHEMA = "comp_struct_soap_lattice_metal.v1"
TOTAL_WIDTH = 483
SOAP_WIDTH = 252
SOAP_DEFAULTS: dict[str, float | int] = {
    "r_cut": 10.0,
    "n_max": 8,
    "l_max": 6,
    "sigma": 1.0,
}
SYMMETRY_PROPERTIES = (
    "spacegroup_num",
    "crystal_system_int",
    "is_centrosymmetric",
)
DENSITY_PROPERTIES = ("density", "vpa", "packing fraction")

CRYSTAL_SYSTEMS = {
    "triclinic": 0,
    "monoclinic": 1,
    "orthorhombic": 2,
    "tetragonal": 3,
    "trigonal": 4,
    "hexagonal": 5,
    "cubic": 6,
}
SYSTEM_ABBREVIATIONS = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "hexagonal": "h",
    "cubic": "c",
}
BRAVAIS = {
    "aP": 0,
    "mP": 1,
    "mC": 2,
    "oP": 3,
    "oC": 4,
    "oI": 5,
    "oF": 6,
    "tP": 7,
    "tI": 8,
    "hP": 9,
    "hR": 10,
    "cP": 11,
    "cI": 12,
    "cF": 13,
}


def normalised_composition(structure: Structure) -> Composition:
    """Return the IUPAC-normalised composition used by the source pipeline."""
    reduced = Composition(structure.formula).get_integer_formula_and_factor()[0]
    return Composition(Composition(reduced).iupac_formula)


def _clean(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, copy=True, nan=0.0, posinf=None, neginf=None)


def _imputing(featurizer_class: type) -> object:
    try:
        return featurizer_class(impute_nan=True)
    except TypeError:
        return featurizer_class()


@lru_cache(maxsize=1)
def _composition_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer(
        [
            ElementProperty.from_preset("magpie", impute_nan=True),
            _imputing(Stoichiometry),
            _imputing(ValenceOrbital),
        ]
    )


@lru_cache(maxsize=1)
def _structure_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer(
        [
            GlobalSymmetryFeatures(list(SYMMETRY_PROPERTIES)),
            DensityFeatures(list(DENSITY_PROPERTIES)),
        ]
    )


@lru_cache(maxsize=8)
def _soap_descriptor(r_cut: float, n_max: int, l_max: int, sigma: float) -> SOAP:
    return SOAP(
        species=["X"],
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic=True,
        sparse=False,
    )


def composition_block(structures: Sequence[Structure]) -> np.ndarray:
    """Return the 146 composition descriptors in their historical order."""
    featurizer = _composition_featurizer()
    return _clean(
        np.asarray(
            [
                featurizer.featurize(normalised_composition(structure))
                for structure in structures
            ],
            dtype=float,
        )
    )


def structure_block(structures: Sequence[Structure]) -> np.ndarray:
    """Return three symmetry and three density descriptors."""
    featurizer = _structure_featurizer()
    rows = np.zeros((len(structures), 6))
    for index, structure in enumerate(structures):
        try:
            rows[index, :] = featurizer.featurize(structure)
        except Exception as error:  # source contract uses an all-zero fallback
            warnings.warn(
                f"structure descriptors failed for {structure.formula}; "
                f"using zeros ({error})",
                stacklevel=2,
            )
    return _clean(rows)


def soap_block(
    structures: Sequence[Structure], parameters: Mapping[str, float | int]
) -> np.ndarray:
    """Return 252 composition-agnostic averaged SOAP descriptors."""
    descriptor = _soap_descriptor(
        float(parameters["r_cut"]),
        int(parameters["n_max"]),
        int(parameters["l_max"]),
        float(parameters["sigma"]),
    )
    if descriptor.get_number_of_features() != SOAP_WIDTH:
        raise ValueError(
            "SOAP parameters produce "
            f"{descriptor.get_number_of_features()} columns; expected {SOAP_WIDTH}"
        )
    rows = np.zeros((len(structures), SOAP_WIDTH))
    for index, structure in enumerate(structures):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.set_chemical_symbols(["X"] * len(atoms))
        rows[index, :] = descriptor.create(atoms).mean(axis=0)
    return _clean(rows)


def lattice_block(structures: Sequence[Structure]) -> np.ndarray:
    """Return direct/reciprocal cell and symmetry identifiers (15 columns)."""
    rows = np.zeros((len(structures), 15))
    for index, structure in enumerate(structures):
        try:
            lattice = structure.lattice
            analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
            system = analyzer.get_crystal_system()
            centering = analyzer.get_space_group_symbol()[0]
            rows[index, :] = [
                *lattice.abc,
                *lattice.angles,
                *lattice.reciprocal_lattice.abc,
                *lattice.reciprocal_lattice.angles,
                CRYSTAL_SYSTEMS[system],
                BRAVAIS.get(SYSTEM_ABBREVIATIONS[system] + centering, -1),
                analyzer.get_space_group_number(),
            ]
        except Exception as error:  # source contract uses an all-zero fallback
            warnings.warn(
                f"lattice descriptors failed for {structure.formula}; "
                f"using zeros ({error})",
                stacklevel=2,
            )
    return _clean(rows)


def metallicity_block(
    structures: Sequence[Structure], checkpoint: Path, atom_init: Path
) -> np.ndarray:
    """Return the frozen metallicity CGCNN representation (64 columns)."""
    representation = crystal_representations(
        list(structures), checkpoint=checkpoint, atom_init=atom_init
    )
    return _clean(representation.detach().numpy().astype(float))


def _settings(parameters: Mapping[str, Any]) -> tuple[dict[str, float | int], int]:
    unknown = sorted(set(parameters) - {"soap", "batch_size"})
    if unknown:
        raise ValueError(f"unknown QRF95 feature parameter(s): {', '.join(unknown)}")
    raw_soap = parameters.get("soap", {})
    if not isinstance(raw_soap, dict):
        raise ValueError("features.parameters.soap must be a table")
    unknown_soap = sorted(set(raw_soap) - set(SOAP_DEFAULTS))
    if unknown_soap:
        raise ValueError(f"unknown QRF95 SOAP parameter(s): {', '.join(unknown_soap)}")
    soap = {**SOAP_DEFAULTS, **raw_soap}
    for name in ("r_cut", "sigma"):
        value = soap[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"features.parameters.soap.{name} must be positive")
    for name in ("n_max", "l_max"):
        value = soap[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                f"features.parameters.soap.{name} must be a positive integer"
            )
    batch_size = parameters.get("batch_size", 32)
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("features.parameters.batch_size must be a positive integer")
    return soap, batch_size


def _column_names() -> tuple[str, ...]:
    names = [f"composition_{index}" for index in range(146)]
    names += [f"symmetry_{name}" for name in SYMMETRY_PROPERTIES]
    names += [f"density_{name.replace(' ', '_')}" for name in DENSITY_PROPERTIES]
    names += [f"soap_{index}" for index in range(SOAP_WIDTH)]
    names += [
        "lattice_a",
        "lattice_b",
        "lattice_c",
        "lattice_alpha",
        "lattice_beta",
        "lattice_gamma",
        "reciprocal_a",
        "reciprocal_b",
        "reciprocal_c",
        "reciprocal_alpha",
        "reciprocal_beta",
        "reciprocal_gamma",
        "crystal_system_id",
        "bravais_id",
        "spacegroup_number",
    ]
    names += [f"metallicity_{index}" for index in range(REPRESENTATION_WIDTH)]
    if len(names) != TOTAL_WIDTH:
        raise AssertionError(f"QRF95 feature names have width {len(names)}")
    return tuple(names)


def feature_rows(
    structures: Sequence[Structure],
    *,
    soap: Mapping[str, float | int],
    metallicity_checkpoint: Path,
    metallicity_atom_init: Path,
) -> np.ndarray:
    """Return the 483 columns for a batch of structures, in contract order.

    Training and inference both come through here, so the block order is
    defined once. A width other than :data:`TOTAL_WIDTH` is a defect in a
    block, not a configuration error, and is reported with the block widths.
    """
    blocks = (
        composition_block(structures),
        structure_block(structures),
        soap_block(structures, dict(soap)),
        lattice_block(structures),
        metallicity_block(structures, metallicity_checkpoint, metallicity_atom_init),
    )
    matrix = np.concatenate(blocks, axis=1)
    if matrix.shape[1] != TOTAL_WIDTH:
        widths = ", ".join(str(block.shape[1]) for block in blocks)
        raise ValueError(
            f"QRF95 feature vector is {matrix.shape[1]} wide; expected "
            f"{TOTAL_WIDTH}. Block widths: {widths}"
        )
    return matrix


def column_names() -> tuple[str, ...]:
    """Return the 483 column names, in contract order."""
    return _column_names()


def resolve_soap(parameters: Mapping[str, Any]) -> dict[str, float | int]:
    """Return the SOAP settings a protocol selected, defaults filled in."""
    soap, _ = _settings(parameters)
    return soap


def build(
    protocol: TrainingProtocol,
    snapshot: Snapshot,
    artifacts: Mapping[str, Path],
) -> FeatureMatrix:
    """Build all 483 columns in bounded-memory structure batches."""
    soap, batch_size = _settings(protocol.features.parameters)
    required = ("metallicity_checkpoint", "metallicity_atom_init")
    missing = [name for name in required if name not in artifacts]
    if missing:
        raise ValueError(
            "the QRF95 feature contract needs artifact(s): " + ", ".join(missing)
        )

    rows: dict[str, tuple[float, ...]] = {}
    for start in range(0, len(snapshot.samples), batch_size):
        samples = snapshot.samples[start : start + batch_size]
        structures = []
        for sample in samples:
            if sample.structure_path is None:
                raise ValueError(f"{sample.sample_id} has no structure file")
            structures.append(Structure.from_file(sample.structure_path))
        matrix = feature_rows(
            structures,
            soap=soap,
            metallicity_checkpoint=artifacts["metallicity_checkpoint"],
            metallicity_atom_init=artifacts["metallicity_atom_init"],
        )
        rows.update(
            {
                sample.sample_id: tuple(float(value) for value in row)
                for sample, row in zip(samples, matrix, strict=True)
            }
        )
    return FeatureMatrix(columns=_column_names(), rows=rows)


register_feature_contract(SCHEMA, build)
