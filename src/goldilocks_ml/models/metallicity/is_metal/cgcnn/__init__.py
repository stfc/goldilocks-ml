"""The metallicity classifier this repository trains and owns.

Fitted from a sealed snapshot, so its record states the dataset, the split, and
the accuracy that the published artifact does not.
"""

from __future__ import annotations

from goldilocks_ml.models.metallicity.is_metal.cgcnn.graphs import (
    SCHEMA,
    clear_graph_cache,
    graphs_for,
)
from goldilocks_ml.models.metallicity.is_metal.cgcnn.trainer import (
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
