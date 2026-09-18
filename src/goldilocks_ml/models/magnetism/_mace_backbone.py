"""Pool a frozen mMACE foundation model into one structure embedding.

Ported from ``2-research/2-mace/1-magnetic-mace/scripts/magnetic_classifier_model.py``
(``MACEEmbedder``, ``safe_load``, ``make_embedder``), with one change: this
module takes a ``pymatgen.core.structure.Structure`` directly rather than
leaving every caller to build its own ASE ``Atoms``, so the feature contract,
the predictor, and any future consumer of this embedding all see one seam.

Needs the ``magnetism`` extra (``mace``, ``e3nn``, ``sphericart``, ``ase``).
Nothing at import time of *this* file touches those packages -- only calling
:func:`embed_structures` does, so importing this module to read its docstring
or constants never requires them, and a missing one surfaces through
:mod:`goldilocks_ml.registry`'s friendly error naming the ``magnetism`` extra
rather than a bare ``ModuleNotFoundError`` here.

The probe is deliberately uninformative: every call runs the backbone once at
an all-zero magnetic moment, with SCF relaxation switched off (this is a
classification embedding, not the magnetic-moment relaxation in
:mod:`goldilocks_ml.models.magnetism.magnetic_moments`, which shares this same
backbone-loading machinery but turns SCF on). No element-specific magnetic
guess and no DFT label leak into the embedding through the probe itself.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

# 128 scalar channels from the backbone's last product-basis block, pooled
# three ways (mean, max, population std) over the atoms in the structure.
EMBEDDING_WIDTH = 384


@contextmanager
def _default_dtype_float64() -> Iterator[None]:
    """Run a block with torch's global default dtype set to float64.

    The backbone needs float64 for numerical stability (its checkpoint was
    trained that way), but ``torch.set_default_dtype`` is process-wide with
    no scoping of its own -- unlike ``safe_load``'s own ``torch.jit.load``
    monkeypatch below, ``MagneticMACECalculator(default_dtype="float64")``
    sets it and leaves it set. Confirmed empirically: without this, any
    float32-assuming model predicting later in the same process (e.g.
    goldilocks_ml's own CGCNN classifier) starts building float64 tensors
    against its float32 weights and fails on the first matmul.
    """
    import torch

    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def safe_load(path: Path, device: str = "cpu") -> Any:
    """Load a whole pickled mMACE model, forcing embedded TorchScript onto device.

    The checkpoint embeds TorchScript sub-modules that ``torch.jit.load``
    would otherwise place on whatever device it was originally saved from.
    This monkeypatches it for the duration of this one load only, then
    restores it -- other code in the same process that calls ``torch.jit.load``
    is unaffected once this returns.
    """
    import torch

    original_jit_load = torch.jit.load
    torch.jit.load = lambda f, map_location=None, **kwargs: original_jit_load(
        f, map_location=device, **kwargs
    )
    try:
        return torch.load(path, map_location=device, weights_only=False)
    finally:
        torch.jit.load = original_jit_load


class MACEEmbedder:
    """Pool one non-SCF, all-zero-moment forward pass into a structure vector."""

    def __init__(self, raw_model: Any, calculator: Any) -> None:
        self.raw_model = raw_model
        self.calculator = calculator
        self._buffer: dict[str, Any] = {}

    def _hook(self, module: Any, inputs: Any, output: Any) -> None:
        features = output[0] if isinstance(output, tuple) else output
        self._buffer["features"] = features.detach().cpu()

    def embed(self, atoms: Any) -> np.ndarray:
        """Return the pooled embedding for one ASE ``Atoms``."""
        import torch

        atoms = atoms.copy()
        moments = np.zeros((len(atoms), 3), dtype=float)
        atoms.arrays["dft_magmom"] = moments

        self._buffer.clear()
        handle = self.raw_model.products[-1].register_forward_hook(self._hook)
        try:
            with _default_dtype_float64():
                # Force a fresh forward pass even when the same Atoms object
                # is embedded twice: ASE calculators otherwise skip
                # recomputation for a structure they consider unchanged
                # since the last call.
                self.calculator.reset()
                atoms.calc = self.calculator
                atoms.get_potential_energy()
        finally:
            handle.remove()

        features = self._buffer.get("features")
        if features is None:
            raise RuntimeError("the product-layer hook produced no embedding")
        # Mean alone can hide a rare magnetic atom in a large cell. All three
        # reductions are permutation invariant and reuse the same forward pass.
        pooled = torch.cat(
            [
                features.mean(dim=0),
                features.max(dim=0).values,
                features.std(dim=0, unbiased=False),
            ]
        )
        return pooled.numpy().astype(np.float32)


@lru_cache(maxsize=4)
def load_embedder(checkpoint: Path, device: str = "cpu") -> MACEEmbedder:
    """Load the frozen backbone once per checkpoint path and device."""
    import torch
    from mace.calculators.mace import MagneticMACECalculator
    from mace.modules import MagneticSCFMACE

    torch.serialization.add_safe_globals([slice])
    raw_model = safe_load(Path(checkpoint), device)
    # A single, non-SCF forward pass: this embedding never relaxes moments, it
    # only probes the frozen backbone once at zero moment.
    one_step_model = MagneticSCFMACE(
        raw_model, n_scf_step=1, use_scf=False, scf_tol=1e-4, scf_logging=False
    )
    with _default_dtype_float64():
        calculator = MagneticMACECalculator(
            models=[one_step_model], device=device, default_dtype="float64"
        )
    return MACEEmbedder(raw_model, calculator)


def embed_structures(
    structures: Sequence[Structure], *, checkpoint: Path, device: str = "cpu"
) -> np.ndarray:
    """Return one embedding row per structure, in input order.

    Shape ``(len(structures), EMBEDDING_WIDTH)``. The backbone is loaded once
    per ``(checkpoint, device)`` pair and cached across calls.
    """
    from pymatgen.io.ase import AseAtomsAdaptor

    if not structures:
        return np.zeros((0, EMBEDDING_WIDTH), dtype=np.float32)
    embedder = load_embedder(Path(checkpoint), device)
    rows = [
        embedder.embed(AseAtomsAdaptor.get_atoms(structure)) for structure in structures
    ]
    return np.stack(rows).astype(np.float32)
