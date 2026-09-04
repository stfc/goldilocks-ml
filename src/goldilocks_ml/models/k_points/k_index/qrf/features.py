"""The 174-column composition/structure/lattice/reciprocal contract.

The block definitions and order match ``goldilocks_core.ml.kindex.features``
as of the initial k-index inference implementation.  They live here because
feature production is part of the trained artifact's versioned contract; Core
will consume the released model rather than remain the owner of featurisation.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty,
    Stoichiometry,
    ValenceOrbital,
)
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
from pymatgen.core.structure import Structure

from goldilocks_ml.protocol import TrainingProtocol
from goldilocks_ml.registry import FeatureMatrix, register_feature_contract
from goldilocks_ml.snapshot import Snapshot

SCHEMA = "cslr.v1"
COMPOSITION_WIDTH = 146
STRUCTURE_WIDTH = 7
LATTICE_WIDTH = 7
RECIPROCAL_WIDTH = 14
TOTAL_WIDTH = 174

LATTICE_COLUMNS = ("a", "b", "c", "alpha", "beta", "gamma", "volume")
RECIPROCAL_COLUMNS = (
    "recip_b1",
    "recip_b2",
    "recip_b3",
    "recip_volume",
    "recip_alpha",
    "recip_beta",
    "recip_gamma",
    "G_tr",
    "G_tr2",
    "G_det",
    "G_cond",
    "bmax_over_bmin",
    "bmid_over_bmin",
    "recip_orthogonality",
)


@lru_cache(maxsize=1)
def _composition_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer(
        [
            ElementProperty.from_preset("magpie", impute_nan=True),
            Stoichiometry(),
            ValenceOrbital(impute_nan=True),
        ]
    )


@lru_cache(maxsize=1)
def _structure_featurizer() -> MultipleFeaturizer:
    return MultipleFeaturizer([GlobalSymmetryFeatures(), DensityFeatures()])


def composition_block(structures: Sequence[Structure]) -> np.ndarray:
    """Return the 146 Magpie/stoichiometry/valence descriptors."""
    featurizer = _composition_featurizer()
    return np.asarray(
        [featurizer.featurize(structure.composition) for structure in structures],
        dtype=float,
    )


def structure_block(structures: Sequence[Structure]) -> np.ndarray:
    """Return numeric global-symmetry and density descriptors.

    Matminer cannot calculate packing fraction for elements whose tabulated
    atomic radius is absent (notably noble gases).  The historical QRF feature
    contract handles a failed structure-descriptor row as zeros; CSLR keeps the
    same deterministic fallback so one such element cannot discard a labelled
    structure at training time or crash inference later.
    """
    featurizer = _structure_featurizer()
    names = featurizer.feature_labels()
    keep = [index for index, name in enumerate(names) if name != "crystal_system"]
    rows = np.zeros((len(structures), STRUCTURE_WIDTH), dtype=float)
    for row, structure in enumerate(structures):
        try:
            values = featurizer.featurize(structure)
            rows[row, :] = [values[index] for index in keep]
        except Exception as error:
            warnings.warn(
                f"CSLR structure descriptors failed for {structure.formula}; "
                f"using zeros ({error})",
                stacklevel=2,
            )
    return rows


def lattice_block(structures: Sequence[Structure]) -> np.ndarray:
    """Return direct-cell lengths, angles, and volume."""
    return np.asarray(
        [
            [
                structure.lattice.a,
                structure.lattice.b,
                structure.lattice.c,
                structure.lattice.alpha,
                structure.lattice.beta,
                structure.lattice.gamma,
                structure.lattice.volume,
            ]
            for structure in structures
        ],
        dtype=float,
    )


def reciprocal_block(structures: Sequence[Structure]) -> np.ndarray:
    """Return reciprocal-cell dimensions, metric invariants, and anisotropy."""
    rows: list[list[float]] = []
    for structure in structures:
        reciprocal = structure.lattice.reciprocal_lattice
        matrix = np.asarray(reciprocal.matrix, dtype=float)
        metric = matrix.T @ matrix
        eigenvalues = np.linalg.eigvalsh(metric)
        lengths = np.sort([reciprocal.a, reciprocal.b, reciprocal.c])
        minimum, middle, maximum = lengths
        cosines = np.cos(
            np.deg2rad([reciprocal.alpha, reciprocal.beta, reciprocal.gamma])
        )
        rows.append(
            [
                reciprocal.a,
                reciprocal.b,
                reciprocal.c,
                reciprocal.volume,
                reciprocal.alpha,
                reciprocal.beta,
                reciprocal.gamma,
                float(np.trace(metric)),
                float(np.trace(metric @ metric)),
                float(np.linalg.det(metric)),
                float(eigenvalues.max() / eigenvalues.min()),
                float(maximum / minimum),
                float(middle / minimum),
                float(np.abs(cosines).sum()),
            ]
        )
    return np.asarray(rows, dtype=float)


def column_names() -> tuple[str, ...]:
    composition = tuple(_composition_featurizer().feature_labels())
    structure = tuple(
        name
        for name in _structure_featurizer().feature_labels()
        if name != "crystal_system"
    )
    columns = composition + structure + LATTICE_COLUMNS + RECIPROCAL_COLUMNS
    widths = (
        len(composition),
        len(structure),
        len(LATTICE_COLUMNS),
        len(RECIPROCAL_COLUMNS),
    )
    expected = (
        COMPOSITION_WIDTH,
        STRUCTURE_WIDTH,
        LATTICE_WIDTH,
        RECIPROCAL_WIDTH,
    )
    if widths != expected or len(columns) != TOTAL_WIDTH:
        raise AssertionError(
            f"CSLR block widths are {widths}; expected {expected} ({TOTAL_WIDTH} total)"
        )
    return columns


def feature_rows(structures: Sequence[Structure]) -> np.ndarray:
    """Return finite CSLR rows in the single contract order."""
    blocks = (
        composition_block(structures),
        structure_block(structures),
        lattice_block(structures),
        reciprocal_block(structures),
    )
    matrix = np.concatenate(blocks, axis=1)
    if matrix.shape != (len(structures), TOTAL_WIDTH):
        widths = tuple(block.shape[1] for block in blocks)
        raise ValueError(f"CSLR block widths are {widths}; expected 146, 7, 7, 14")
    if not np.isfinite(matrix).all():
        row, column = np.argwhere(~np.isfinite(matrix))[0]
        raise ValueError(
            f"CSLR produced a non-finite value at row {row}, column "
            f"{column_names()[int(column)]!r}"
        )
    return matrix


def _batch_size(parameters: Mapping[str, Any]) -> int:
    unknown = sorted(set(parameters) - {"batch_size"})
    if unknown:
        raise ValueError(f"unknown CSLR feature parameter(s): {', '.join(unknown)}")
    batch_size = parameters.get("batch_size", 128)
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("features.parameters.batch_size must be a positive integer")
    return batch_size


def build(
    protocol: TrainingProtocol,
    snapshot: Snapshot,
    artifacts: Mapping[str, Path],
) -> FeatureMatrix:
    """Build CSLR features for every verified snapshot structure."""
    if artifacts:
        raise ValueError("the CSLR feature contract has no artifact dependencies")
    batch_size = _batch_size(protocol.features.parameters)
    rows: dict[str, tuple[float, ...]] = {}
    for start in range(0, len(snapshot.samples), batch_size):
        samples = snapshot.samples[start : start + batch_size]
        missing = [
            sample.sample_id for sample in samples if sample.structure_path is None
        ]
        if missing:
            raise ValueError(
                f"the CSLR feature contract needs structures; missing {missing[0]}"
            )
        structures = [
            Structure.from_file(sample.structure_path)  # type: ignore[arg-type]
            for sample in samples
        ]
        matrix = feature_rows(structures)
        rows.update(
            {
                sample.sample_id: tuple(float(value) for value in matrix[index])
                for index, sample in enumerate(samples)
            }
        )
    return FeatureMatrix(columns=column_names(), rows=rows)


register_feature_contract(SCHEMA, build)
