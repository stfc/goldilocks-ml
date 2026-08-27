"""Models, one package per task then per model.

The layout matches ``deposits/`` and the released artifact namespace: a task
folder holds one package per model, and each model package owns its own
protocol, feature contract, and trainer.
"""

from __future__ import annotations
