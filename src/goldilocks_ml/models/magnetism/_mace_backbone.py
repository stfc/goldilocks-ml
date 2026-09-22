"""Pool a frozen mMACE foundation model into one structure embedding.

Ported from ``2-research/2-mace/1-magnetic-mace/scripts/magnetic_classifier_model.py``
(``MACEEmbedder``, ``safe_load``, ``make_embedder``), with one change: this
module takes a ``pymatgen.core.structure.Structure`` directly rather than
leaving every caller to build its own ASE ``Atoms``, so the feature contract,
the predictor, and any future consumer of this embedding all see one seam.

Needs ``mace``, ``e3nn``, ``sphericart`` and ``ase`` -- none of which install
via a `pip`-able extra of this package (see "Use the is_magnetic classifier"
in README.md for the manual steps; ``mace`` specifically is a research
collaborator's fork with no PyPI release at all). Nothing at import time of
*this* file touches those packages -- only calling :func:`embed_structures`
does, so importing this module to read its docstring or constants never
requires them, and a missing one raises a message naming the manual install
steps rather than a bare ``ModuleNotFoundError`` here.

The probe is deliberately uninformative: every call runs the backbone once at
an all-zero magnetic moment, with SCF relaxation switched off (this is a
classification embedding, not the magnetic-moment relaxation in
:mod:`goldilocks_ml.models.magnetism.magnetic_moments`, which shares this same
backbone-loading machinery but turns SCF on). No element-specific magnetic
guess and no DFT label leak into the embedding through the probe itself.
"""

from __future__ import annotations

import threading
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

# Serializes every operation in this module that touches process-wide global
# state or the shared cached `MACEEmbedder` instance: `_default_dtype_float64`
# (three callers -- this module's calculator construction and embed step,
# plus magnetic_moments.fm_fim_relax.relax's), `safe_load`'s `torch.jit.load`
# monkeypatch, and `MACEEmbedder.embed`'s hook registration + buffer. Each of
# those is documented at its own use below; all three share this one lock
# because they are the same underlying problem (mace mutates or shares
# process-wide state with no synchronisation of its own) and goldilocks-core
# is exactly the kind of caller that triggers it -- FastAPI runs handlers in
# a threadpool. Reentrant because `MACEEmbedder.embed` holds it for its whole
# body and then also calls `_default_dtype_float64`, which acquires it again
# on the same thread. Serializing costs little: mace inference is CPU-bound
# anyway.
_mace_lock = threading.RLock()


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

    Without ``_mace_lock``, two threads racing through the save/restore
    below can also interleave: thread B reads the "previous" dtype while it
    has already been changed to float64 by thread A, so when A restores
    first, B's computation is silently corrupted mid-flight, and when B
    then restores last it writes back the *wrong* "previous" value
    (float64), leaving the global stuck exactly like the bug this context
    manager exists to prevent -- confirmed empirically with two threads and
    a barrier forcing that exact interleaving.
    """
    import torch

    with _mace_lock:
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

    Locked for the same reason ``_default_dtype_float64`` is: two threads
    racing through this save/restore of ``torch.jit.load`` (a process-wide
    global with no scoping of its own) can leave it permanently monkeypatched
    to whichever thread's ``device`` lost the race, rather than restored to
    the real original -- confirmed empirically with a forced two-thread
    interleaving.
    """
    import torch

    with _mace_lock:
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
        """Return the pooled embedding for one ASE ``Atoms``.

        The whole body is locked, not just the forward pass: ``load_embedder``
        hands the *same* cached instance to every caller, so ``self._buffer``
        and the hook registered on ``self.raw_model.products[-1]`` are shared
        mutable state across concurrent callers. Locking only the inference
        step still let one thread's stale, not-yet-removed hook fire during
        another thread's forward pass and overwrite the shared buffer before
        the first thread read it back -- silently returning another request's
        embedding instead of raising. Confirmed with a forced two-thread
        interleaving using fake hooks standing in for the real ones.
        """
        import torch

        atoms = atoms.copy()
        moments = np.zeros((len(atoms), 3), dtype=float)
        atoms.arrays["dft_magmom"] = moments

        with _mace_lock:
            self._buffer.clear()
            handle = self.raw_model.products[-1].register_forward_hook(self._hook)
            try:
                with _default_dtype_float64():
                    # Force a fresh forward pass even when the same Atoms
                    # object is embedded twice: ASE calculators otherwise
                    # skip recomputation for a structure they consider
                    # unchanged since the last call.
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

    try:
        from mace.calculators.mace import MagneticMACECalculator
        from mace.modules import MagneticSCFMACE
    except ModuleNotFoundError as error:
        from goldilocks_ml.registry import MAGNETISM_INSTALL_HINT

        raise ValueError(f"is_magnetic needs mace; {MAGNETISM_INSTALL_HINT}") from error

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
