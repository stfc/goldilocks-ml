"""CGCNN representation used by the QRF95 feature contract.

Adapted from ``stfc/goldilocks_kpoints`` (``models/cgcnn.py`` and graph
utilities), © 2024 Science and Technology Facilities Council, CC BY 4.0.
Layer names and shapes are fixed by the published metallicity checkpoint.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as functional
from pymatgen.core.structure import Structure
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing, global_mean_pool

REPRESENTATION_WIDTH = 64
RADIUS = 10.0
MAX_NEIGHBORS = 12


class RBFExpansion(nn.Module):
    """Expand scalar distances over a fixed grid of Gaussian centres."""

    def __init__(
        self,
        vmin: float = 0.0,
        vmax: float = 8.0,
        bins: int = 40,
        lengthscale: float | None = None,
    ) -> None:
        super().__init__()
        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer("centers", centers)
        self.gamma = 1 / ((lengthscale or (centers[1] - centers[0]).item()) ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Return the RBF expansion of each distance."""
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class CGCNNConv(MessagePassing):
    """Apply the gated convolution used by the published checkpoint."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int) -> None:
        super().__init__(aggr="add")
        self.lin_f = nn.Linear(2 * node_dim + edge_dim, out_dim)
        self.lin_s = nn.Linear(2 * node_dim + edge_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Propagate messages along every edge."""
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """Return one gated message."""
        joined = torch.cat([x_i, x_j, edge_attr], dim=1)
        return torch.sigmoid(self.lin_f(joined)) * functional.softplus(
            self.lin_s(joined)
        )

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add the residual and normalise the aggregated messages."""
        return self.batch_norm(aggr_out + x)


class CGCNN(nn.Module):
    """The checkpoint-compatible CGCNN classification architecture."""

    def __init__(
        self,
        orig_atom_fea_len: int = 92,
        edge_feat_dim: int = 64,
        h_fea_len: int = 128,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        n_h: int = 3,
        num_classes: int = 2,
        pooling_type: str = "mean_pool",
        **ignored: object,
    ) -> None:
        super().__init__()
        if pooling_type != "mean_pool":
            raise ValueError(f"unsupported pooling type {pooling_type!r}")
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=edge_feat_dim)
        self.convs = nn.ModuleList(
            CGCNNConv(atom_fea_len, edge_feat_dim, atom_fea_len) for _ in range(n_conv)
        )
        self.conv_to_fc_softplus = nn.Softplus()
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        if n_h > 1:
            self.fcs = nn.ModuleList(
                nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)
            )
            self.softpluses = nn.ModuleList(nn.Softplus() for _ in range(n_h - 1))
        self.fc_out = nn.Linear(h_fea_len, num_classes)

    def extract_crystal_repr(self, data: Data) -> torch.Tensor:
        """Return the 64-wide pooled representation before the dense head."""
        nodes = self.embedding(data.x)
        edges = self.rbf(data.edge_attr.view(-1))
        for convolution in self.convs:
            nodes = convolution(nodes, data.edge_index, edges)
        return global_mean_pool(nodes, data.batch)

    def forward(self, data: Data) -> torch.Tensor:
        """Return metallicity class logits."""
        values = self.conv_to_fc_softplus(self.extract_crystal_repr(data))
        values = self.conv_to_fc_softplus(self.conv_to_fc(values))
        if hasattr(self, "fcs"):
            for layer, activation in zip(self.fcs, self.softpluses, strict=True):
                values = activation(layer(values))
        return self.fc_out(values)


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
    """Build the fixed radius graph expected by the checkpoint."""
    table = load_atom_features(atom_init)
    features = []
    for site in structure:
        feature = table.get(str(site.specie.number))
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


@lru_cache(maxsize=4)
def load_cgcnn(checkpoint: Path) -> CGCNN:
    """Load the digest-verified checkpoint without arbitrary pickle objects."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = CGCNN(**state["hyper_parameters"]["model"])
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
) -> torch.Tensor:
    """Return one pooled representation per structure in input order."""
    model = load_cgcnn(checkpoint)
    graphs = [build_graph(structure, atom_init) for structure in structures]
    with torch.no_grad():
        batch = Batch.from_data_list(graphs)
        return model.extract_crystal_repr(batch)
