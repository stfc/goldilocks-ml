"""The metallicity model's crystal representation, as other models consume it.

The k-mesh feature contract embeds this representation, which is why its
protocol pins this checkpoint's SHA-256. A different checkpoint silently
produces a different feature vector.
"""

from __future__ import annotations

from pathlib import Path

import torch
from pymatgen.core.structure import Structure
from torch_geometric.data import Batch

from goldilocks_ml.models.metallicity.cgcnn.graph import build_graph
from goldilocks_ml.models.metallicity.cgcnn.network import CGCNN

REPRESENTATION_WIDTH = 64


def load_model(checkpoint: Path) -> CGCNN:
    """Load the released checkpoint into an evaluation-mode network."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hyperparameters = state["hyper_parameters"]["model"]
    model = CGCNN(**hyperparameters)
    # Lightning stores the wrapped module, so every key carries a "model." prefix.
    weights = {
        key.removeprefix("model."): value for key, value in state["state_dict"].items()
    }
    model.load_state_dict(weights)
    model.eval()
    return model


def crystal_representations(
    structures: list[Structure],
    *,
    checkpoint: Path,
    atom_init: Path,
    batch_size: int = 32,
) -> torch.Tensor:
    """Return one pooled representation per structure, in the given order."""
    model = load_model(checkpoint)
    graphs = [build_graph(structure, atom_init) for structure in structures]
    chunks = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start : start + batch_size])
            chunks.append(model.extract_crystal_repr(batch))
    return (
        torch.cat(chunks, dim=0) if chunks else torch.empty((0, REPRESENTATION_WIDTH))
    )
