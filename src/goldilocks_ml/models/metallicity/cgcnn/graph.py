"""Turn a crystal structure into the graph the CGCNN checkpoint expects.

Adapted from `stfc/goldilocks_kpoints` (`utils/cgcnn_graph.py`,
`utils/atom_features_utils.py`), (c) 2024 Science and Technology Facilities
Council, CC BY 4.0.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from pymatgen.core.structure import Structure
from torch_geometric.data import Data

RADIUS = 10.0
MAX_NEIGHBORS = 12


@lru_cache(maxsize=4)
def load_atom_features(path: Path) -> dict[str, list[float]]:
    """Load the atomic embedding table keyed by atomic number."""
    with Path(path).open(encoding="utf-8") as handle:
        table = json.load(handle)
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{path} must contain a non-empty JSON object")
    return table


def build_graph(
    structure: Structure,
    atom_init: Path,
    *,
    radius: float = RADIUS,
    max_neighbors: int = MAX_NEIGHBORS,
) -> Data:
    """Build the radius graph for one structure.

    Each atom keeps its ``max_neighbors`` closest neighbours within ``radius``,
    and every edge carries the interatomic distance.
    """
    table = load_atom_features(atom_init)
    features = []
    for site in structure:
        number = str(site.specie.number)
        feature = table.get(number)
        if feature is None:
            raise ValueError(f"no atomic feature for element {site.specie}")
        features.append(feature)

    edges: list[list[int]] = []
    distances: list[list[float]] = []
    for index, neighbors in enumerate(
        structure.get_all_neighbors(radius, include_index=True)
    ):
        for neighbor in sorted(neighbors, key=lambda entry: entry[1])[:max_neighbors]:
            edges.append([index, neighbor[2]])
            distances.append([neighbor[1]])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(distances, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
