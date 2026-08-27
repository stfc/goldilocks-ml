"""The CGCNN architecture, ported so the released checkpoint loads exactly.

Adapted from `stfc/goldilocks_kpoints` (`models/cgcnn.py`), (c) 2024 Science and
Technology Facilities Council, CC BY 4.0. Layer names and shapes must not drift:
they are the keys of the published `is_metal.ckpt` state dict.
"""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool


class RBFExpansion(nn.Module):
    """Expand a scalar distance over a fixed grid of Gaussian centres."""

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
    """The gated convolution from the CGCNN paper.

    ``h_i' = BN(h_i + sum_j sigmoid(z_ij W_f) * softplus(z_ij W_s))`` where
    ``z_ij`` concatenates the two node features and the edge feature.
    """

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
        """Return the gated message for one edge."""
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        return torch.sigmoid(self.lin_f(z)) * functional.softplus(self.lin_s(z))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Add the aggregated messages to the residual and normalise."""
        return self.batch_norm(aggr_out + x)


class CGCNN(nn.Module):
    """Crystal graph convolutional network.

    Only the pieces the released checkpoint needs are kept: the classification
    head and the crystal representation. Regression, quantile, and
    additional-compound-feature variants of the original are not ported.
    """

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
            CGCNNConv(atom_fea_len, edge_feat_dim, out_dim=atom_fea_len)
            for _ in range(n_conv)
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
        """Return mean-pooled node features after the convolutions.

        This is the representation the k-mesh feature contract consumes. It is
        taken before ``conv_to_fc``, so its width is ``atom_fea_len``.
        """
        x = self.embedding(data.x)
        edge_attr = self.rbf(data.edge_attr.view(-1))
        for conv in self.convs:
            x = conv(x, data.edge_index, edge_attr)
        return global_mean_pool(x, data.batch)

    def forward(self, data: Data) -> torch.Tensor:
        """Return class logits."""
        x = self.conv_to_fc_softplus(self.extract_crystal_repr(data))
        x = self.conv_to_fc_softplus(self.conv_to_fc(x))
        if hasattr(self, "fcs"):
            for fc, softplus in zip(self.fcs, self.softpluses, strict=True):
                x = softplus(fc(x))
        return self.fc_out(x)
