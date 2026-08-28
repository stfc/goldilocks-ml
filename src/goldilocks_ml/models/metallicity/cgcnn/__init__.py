"""Crystal graph convolutional network for binary metallicity.

Only the released checkpoint's architecture and its crystal representation live
here, because the k-mesh feature contract consumes that representation. The
training side of this model belongs to its own issue.
"""

from __future__ import annotations

from goldilocks_ml.models.metallicity.cgcnn.representation import (
    CGCNN,
    REPRESENTATION_WIDTH,
    build_graph,
    crystal_representations,
    load_cgcnn,
)

__all__ = [
    "CGCNN",
    "REPRESENTATION_WIDTH",
    "build_graph",
    "crystal_representations",
    "load_cgcnn",
]
