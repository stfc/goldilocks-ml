"""A dependency-free trainer and feature contract used only by the test suite.

These exist so CI can exercise the complete protocol workflow without a GPU,
private data, or the scientific dependency stack. They are not models.
"""

from __future__ import annotations

from pipeline import baseline as _baseline  # noqa: F401  (registers)
from pipeline import tabular as _tabular  # noqa: F401  (registers)
