"""FM/ferrimagnetic-only initial moments, relaxed on a frozen mMACE surface.

Not a ground-state search: one initial guess is relaxed to a local minimum on
the backbone's own potential energy surface, collinear only, on a fixed
(unrelaxed) geometry. Antiferromagnetic initialisation is out of scope -- see
the package docstring.
"""

from __future__ import annotations
