"""The metallicity classifier this repository trains and owns.

Separate from :mod:`goldilocks_ml.models.metallicity.cgcnn`, which ports the
published checkpoint and exists to supply the QRF95 feature contract. This
package fits the same architecture from a sealed snapshot, so its record can
state the dataset, the split, and the accuracy that the published artifact
does not.
"""

from __future__ import annotations

from goldilocks_ml.models.metallicity.cgcnn2.graphs import (
    SCHEMA,
    clear_graph_cache,
    graphs_for,
)
from goldilocks_ml.models.metallicity.cgcnn2.trainer import (
    ARCHITECTURE,
    RUNTIME,
    TRAINER,
    CGCNNClassifier,
)

__all__ = [
    "ARCHITECTURE",
    "RUNTIME",
    "SCHEMA",
    "TRAINER",
    "CGCNNClassifier",
    "clear_graph_cache",
    "graphs_for",
]
