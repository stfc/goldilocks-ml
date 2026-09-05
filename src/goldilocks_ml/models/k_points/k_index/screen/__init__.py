"""Screening for structures that need a dense k-mesh.

Unlike the other models here, this one does not advise a calculation. It ranks
candidate structures by how likely they are to need a mesh at rung 11 or above,
so that machine time spent growing the dataset goes where the labels are scarce.
Its consumer is the acquisition campaign, not Quantum ESPRESSO.
"""

from goldilocks_ml.models.k_points.k_index.screen import (  # noqa: F401
    predictor,
    trainer,
)
