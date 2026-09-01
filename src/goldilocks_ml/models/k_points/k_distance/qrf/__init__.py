"""Quantile random forest over composition, structure, SOAP and lattice features.

:mod:`embedding` holds the published metallicity checkpoint's architecture and
its pooled crystal representation, which this feature contract consumes. It
lives here rather than beside the metallicity trainer because it is part of
this model's feature definition, not a model in its own right.
"""

from __future__ import annotations
