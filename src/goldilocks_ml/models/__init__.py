"""Model families, one package per task then per model.

The layout mirrors ``deposits/`` and the released artifact namespace: a task
folder (``kmesh``, ``metallicity``) holds one package per model, and each model
package owns its own protocol, feature contract, and trainer.
"""

from __future__ import annotations
