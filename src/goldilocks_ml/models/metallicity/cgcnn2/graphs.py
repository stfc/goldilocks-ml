"""Crystal graphs for the trainable metallicity classifier.

A graph network consumes the crystal, not a fixed-width row, so this contract
computes no columns. What it does is assert that the snapshot supplies
structures and that the atomic embedding table the architecture needs is
pinned, then leave the structures for the trainer to read. Declaring it keeps
a protocol honest about what its model eats.

Graph construction itself is shared with the published checkpoint's port, so
both models see a crystal the same way and their results stay comparable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from goldilocks_ml.models.metallicity.cgcnn import build_graph
from goldilocks_ml.registry import FeatureMatrix, register_feature_contract

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from goldilocks_ml.protocol import TrainingProtocol
    from goldilocks_ml.snapshot import Sample, Snapshot

SCHEMA = "crystal_graph.v1"
ATOM_INIT = "atom_init"


def build(
    protocol: TrainingProtocol,
    snapshot: Snapshot,
    artifacts: Mapping[str, Path],
) -> FeatureMatrix:
    """Check what the graph model needs and return no columns."""
    if ATOM_INIT not in artifacts:
        raise ValueError(f"the {SCHEMA} feature contract needs artifact: {ATOM_INIT}")
    missing = [
        sample.sample_id for sample in snapshot.samples if sample.structure_path is None
    ]
    if missing:
        raise ValueError(
            f"{SCHEMA} needs a structure for every sample; {len(missing)} have "
            f"none, starting with {missing[0]}"
        )
    return FeatureMatrix(
        columns=(), rows={sample.sample_id: () for sample in snapshot.samples}
    )


# Building a graph parses a CIF and searches neighbours, which costs far more
# than holding the result. Training and then evaluating four splits would
# otherwise rebuild every crystal twice.
_CACHE: dict[tuple[str, str], Data] = {}


def graphs_for(
    samples: Sequence[Sample], atom_init: Path, *, cache: bool = True
) -> list[Data]:
    """Return one graph per sample, in the order given."""
    from pymatgen.core.structure import Structure

    table = str(Path(atom_init).resolve())
    graphs = []
    for sample in samples:
        if sample.structure_path is None:
            raise ValueError(f"{sample.sample_id} has no structure file")
        key = (str(Path(sample.structure_path).resolve()), table)
        graph = _CACHE.get(key) if cache else None
        if graph is None:
            graph = build_graph(
                Structure.from_file(sample.structure_path), Path(atom_init)
            )
            if cache:
                _CACHE[key] = graph
        graphs.append(graph)
    return graphs


def clear_graph_cache() -> None:
    """Forget every cached graph."""
    _CACHE.clear()


register_feature_contract(SCHEMA, build)
