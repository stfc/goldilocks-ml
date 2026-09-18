"""Models advising magnetism: whether a structure is spin-polarised, how the
spins are arranged, and what to start the moments at.

Only ``is_magnetic`` is served today. ``ordering`` and ``magnetic_moments``
are declared in the ecosystem's vocabulary but have no model behind them yet
-- see :mod:`goldilocks_ml.models.magnetism.magnetic_moments` for the
FM/FiM-only groundwork that exists so far.
"""

from __future__ import annotations
