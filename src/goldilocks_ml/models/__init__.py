"""Models, one package per task then per model.

The layout matches ``deposits/`` and the released artifact namespace: a task
folder holds one package per model, and each model package owns its own
protocol, feature contract, and trainer.
"""

from __future__ import annotations

from goldilocks_ml.models import kmesh as _kmesh  # noqa: F401,E402
from goldilocks_ml.models import metallicity as _metallicity  # noqa: F401,E402
