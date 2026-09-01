"""Models advising the k-point mesh.

Quantities under this parameter -- k-distance, k-index, k-line-density,
k-points-per-atom -- all name a rung on the same ladder of reachable meshes.
Goldilocks Core owns the ladder and the conversion onto it.
"""

from __future__ import annotations
