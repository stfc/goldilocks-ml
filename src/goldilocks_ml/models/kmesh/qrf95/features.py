"""The `comp_struct_soap_lattice_metal` feature contract.

Reproduces the 483-column feature vector the released QRF95 model was fitted
on. The block order and every width is fixed by that model: it reports
`n_features_in_ = 483`, and the decomposition below is the only one that
reaches it. See README.md for the table.

Adapted from `stfc/goldilocks_kpoints` (`utils/compound_features_utils.py`),
(c) 2024 Science and Technology Facilities Council, CC BY 4.0.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

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

from goldilocks_ml.models.metallicity.cgcnn.embedding import (
    REPRESENTATION_WIDTH,
    crystal_representations,
)
from goldilocks_ml.protocol import TrainingProtocol
from goldilocks_ml.registry import FeatureMatrix, register_feature_contract
from goldilocks_ml.snapshot import Snapshot

SCHEMA = "comp_struct_soap_lattice_metal"
TOTAL_WIDTH = 483
SOAP_DEFAULTS = {"r_cut": 10.0, "n_max": 8, "l_max": 6, "sigma": 1.0}
SYMMETRY_PROPERTIES = ["spacegroup_num", "crystal_system_int", "is_centrosymmetric"]
DENSITY_PROPERTIES = ["density", "vpa", "packing fraction"]

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
    """Return the IUPAC-normalised composition the reference implementation used."""
    reduced = Composition(structure.formula).get_integer_formula_and_factor()[0]
    return Composition(Composition(reduced).iupac_formula)


def _clean(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, copy=True, nan=0.0, posinf=None, neginf=None)


def _imputing(featurizer_class: type) -> object:
    """Build a featurizer with NaN imputation where it supports the option.

    ``Stoichiometry`` does not take ``impute_nan``; the reference implementation
    falls back to the plain constructor, and the widths depend on it.
    """
    try:
        return featurizer_class(impute_nan=True)
    except TypeError:
        return featurizer_class()


def composition_block(structures: Sequence[Structure]) -> np.ndarray:
    """Blocks 1 to 3: matminer composition descriptors (132 + 6 + 8)."""
    featurizer = MultipleFeaturizer(
        [
            ElementProperty.from_preset("magpie", impute_nan=True),
            _imputing(Stoichiometry),
            _imputing(ValenceOrbital),
        ]
    )
    rows = [
        featurizer.featurize(normalised_composition(structure))
        for structure in structures
    ]
    return _clean(np.asarray(rows, dtype=float))


def structure_block(structures: Sequence[Structure]) -> np.ndarray:
    """Blocks 4 and 5: symmetry and density descriptors (3 + 3).

    ``GlobalSymmetryFeatures`` takes three named properties, not its
    five-property default.
    """
    featurizer = MultipleFeaturizer(
        [
            GlobalSymmetryFeatures(SYMMETRY_PROPERTIES),
            DensityFeatures(DENSITY_PROPERTIES),
        ]
    )
    width = len(SYMMETRY_PROPERTIES) + len(DENSITY_PROPERTIES)
    rows = np.zeros((len(structures), width))
    for index, structure in enumerate(structures):
        try:
            rows[index, :] = featurizer.featurize(structure)
        except Exception:  # noqa: BLE001 - the reference falls back to zeros
            warnings.warn(
                f"structure features failed for {structure.formula}; using zeros",
                stacklevel=2,
            )
    return _clean(rows)


def soap_block(
    structures: Sequence[Structure], parameters: Mapping[str, float]
) -> np.ndarray:
    """Block 6: SOAP with every species collapsed to one type (252)."""
    descriptor = SOAP(
        species=["X"],
        r_cut=float(parameters["r_cut"]),
        n_max=int(parameters["n_max"]),
        l_max=int(parameters["l_max"]),
        sigma=float(parameters["sigma"]),
        periodic=True,
        sparse=False,
    )
    rows = np.zeros((len(structures), descriptor.get_number_of_features()))
    for index, structure in enumerate(structures):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.set_chemical_symbols(["X"] * len(atoms))
        rows[index, :] = descriptor.create(atoms).mean(axis=0)
    return _clean(rows)


def lattice_block(structures: Sequence[Structure]) -> np.ndarray:
    """Block 7: direct and reciprocal cell, plus symmetry identifiers (15)."""
    rows = np.zeros((len(structures), 15))
    for index, structure in enumerate(structures):
        try:
            lattice = structure.lattice
            values = [
                *lattice.abc,
                *lattice.angles,
                *lattice.reciprocal_lattice.abc,
                *lattice.reciprocal_lattice.angles,
            ]
            analyzer = SpacegroupAnalyzer(structure, symprec=0.01)
            system = analyzer.get_crystal_system()
            centering = analyzer.get_space_group_symbol()[0]
            values.append(CRYSTAL_SYSTEMS[system])
            values.append(BRAVAIS.get(SYSTEM_ABBREVIATIONS[system] + centering, -1))
            values.append(analyzer.get_space_group_number())
            rows[index, :] = values
        except Exception:  # noqa: BLE001 - the reference falls back to zeros
            warnings.warn(
                f"lattice features failed for {structure.formula}; using zeros",
                stacklevel=2,
            )
    return _clean(rows)


def metallicity_block(
    structures: Sequence[Structure], checkpoint: Path, atom_init: Path
) -> np.ndarray:
    """Block 8: the metallicity model's crystal representation (64)."""
    representation = crystal_representations(
        list(structures), checkpoint=checkpoint, atom_init=atom_init
    )
    return _clean(representation.numpy().astype(float))


def build(
    protocol: TrainingProtocol,
    snapshot: Snapshot,
    artifacts: Mapping[str, Path],
) -> FeatureMatrix:
    """Assemble the 483-column feature matrix for a whole snapshot."""
    unknown = sorted(set(protocol.features.parameters) - {"soap"})
    if unknown:
        raise ValueError(f"unknown feature parameter(s): {', '.join(unknown)}")
    soap_parameters = {**SOAP_DEFAULTS, **protocol.features.parameters.get("soap", {})}

    for name in ("metallicity_checkpoint", "metallicity_atom_init"):
        if name not in artifacts:
            raise ValueError(f"the feature contract needs the {name} artifact")

    structures = []
    for sample in snapshot.samples:
        if sample.structure_path is None:
            raise ValueError(f"{sample.sample_id} has no structure file")
        structures.append(Structure.from_file(sample.structure_path))

    blocks = [
        composition_block(structures),
        structure_block(structures),
        soap_block(structures, soap_parameters),
        lattice_block(structures),
        metallicity_block(
            structures,
            artifacts["metallicity_checkpoint"],
            artifacts["metallicity_atom_init"],
        ),
    ]
    matrix = np.concatenate(blocks, axis=1)
    if matrix.shape[1] != TOTAL_WIDTH:
        widths = ", ".join(str(block.shape[1]) for block in blocks)
        raise ValueError(
            f"the feature vector is {matrix.shape[1]} wide; the released model "
            f"expects {TOTAL_WIDTH}. Block widths: {widths}"
        )

    columns = tuple(_column_names(len(blocks[2][0])))
    return FeatureMatrix(
        columns=columns,
        rows={
            sample_id: tuple(row)
            for sample_id, row in zip(snapshot.sample_ids, matrix, strict=True)
        },
    )


def _column_names(soap_width: int) -> list[str]:
    """Name every column so a run bundle records what the model was fitted on."""
    names = [f"composition_{index}" for index in range(146)]
    names += [f"symmetry_{name}" for name in SYMMETRY_PROPERTIES]
    names += [f"density_{name.replace(' ', '_')}" for name in DENSITY_PROPERTIES]
    names += [f"soap_{index}" for index in range(soap_width)]
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
    return names


register_feature_contract(SCHEMA, build)
