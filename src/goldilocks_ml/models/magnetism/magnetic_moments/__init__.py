"""Groundwork for the ``ordering`` and ``magnetic_moments`` quantities.

No model is published from this package yet -- see
:mod:`goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax` for what
exists: an automatic FM/ferrimagnetic initial-moment guess and a ported
moment relaxation on a frozen mMACE potential energy surface. Antiferromagnetic
sublattice detection is not implemented (there is no code path anywhere in the
project this was ported from that assigns opposite signs to symmetry-equivalent
sites of the same element), and the contract mechanism this would need
(``ContractSpec.labels`` for ``ordering``, ``ContractSpec.index_convention``
for per-site ``magnetic_moments``) is built in
:mod:`goldilocks_ml.inference` but has no released model behind it.
"""

from __future__ import annotations
