"""The 384-column frozen mMACE embedding feature contract.

Ported from ``2-research/2-mace/1-magnetic-mace/scripts/magnetic_classifier_model.py``
(``MACEEmbedder``). Unlike the CGCNN graph contract, this one *does* produce
fixed-width columns -- the backbone's pooled representation is a plain vector,
not a variable-size graph -- so it follows the QRF95 feature contract's shape
instead: real column names, checked against the fitted model's own record at
load time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from goldilocks_ml.models.magnetism._mace_backbone import (
    EMBEDDING_WIDTH,
    embed_structures,
)
from goldilocks_ml.registry import FeatureMatrix, register_feature_contract

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

    from goldilocks_ml.protocol import TrainingProtocol
    from goldilocks_ml.snapshot import Snapshot

SCHEMA = "mace_probe_embedding.v1"
MACE_BACKBONE = "mace_backbone"


def column_names() -> tuple[str, ...]:
    """Return the embedding's column names, in contract order."""
    return tuple(f"mace_probe_{index}" for index in range(EMBEDDING_WIDTH))


def feature_rows(
    structures: Sequence[Structure], *, checkpoint: Path, batch_size: int = 32
) -> list[tuple[float, ...]]:
    """Return one 384-wide row per structure, batched to bound memory."""
    rows: list[tuple[float, ...]] = []
    for start in range(0, len(structures), batch_size):
        window = structures[start : start + batch_size]
        matrix = embed_structures(window, checkpoint=checkpoint)
        rows.extend(tuple(float(value) for value in row) for row in matrix)
    return rows


def build(
    protocol: TrainingProtocol,
    snapshot: Snapshot,
    artifacts: Mapping[str, Path],
) -> FeatureMatrix:
    """Build all 384 columns in bounded-memory structure batches."""
    from pymatgen.core.structure import Structure

    if MACE_BACKBONE not in artifacts:
        raise ValueError(
            f"the {SCHEMA} feature contract needs artifact: {MACE_BACKBONE}"
        )
    checkpoint = artifacts[MACE_BACKBONE]

    columns = column_names()
    rows: dict[str, tuple[float, ...]] = {}
    batch_size = 32
    for start in range(0, len(snapshot.samples), batch_size):
        samples = snapshot.samples[start : start + batch_size]
        structures = []
        for sample in samples:
            if sample.structure_path is None:
                raise ValueError(f"{sample.sample_id} has no structure file")
            structures.append(Structure.from_file(sample.structure_path))
        matrix = feature_rows(structures, checkpoint=checkpoint, batch_size=batch_size)
        ids = (sample.sample_id for sample in samples)
        rows.update(dict(zip(ids, matrix, strict=True)))
    return FeatureMatrix(columns=columns, rows=rows)


register_feature_contract(SCHEMA, build)
