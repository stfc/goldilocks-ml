"""Pool a frozen mMACE foundation model into one structure embedding.

Ported from ``2-research/2-mace/1-magnetic-mace/scripts/magnetic_classifier_model.py``
(``MACEEmbedder``, ``safe_load``, ``make_embedder``), with one change: this
module takes a ``pymatgen.core.structure.Structure`` directly rather than
leaving every caller to build its own ASE ``Atoms``, so the feature contract,
the predictor, and any future consumer of this embedding all see one seam.

Runs on ``mace-torch[magnetic]``, a dependency of this package. Nothing at
import time of *this* file touches mace -- only calling
:func:`embed_structures` does, so importing this module to read its constants
stays cheap.

The probe is deliberately uninformative: every call runs the backbone forward
once at an all-zero magnetic moment, with no SCF relaxation and no gradients
(this is a classification embedding, not the magnetic-moment relaxation in
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

# The `Atoms.arrays` key both this module and fm_fim_relax put the input
# moments under when building the backbone's batch.
MAGMOM_KEY = "dft_magmom"

# Serializes every operation in this module that touches process-wide global
# state or the shared cached `MACEEmbedder` instance: `_default_dtype_float64`
# (used by `_zero_moment_forward` here and by
# magnetic_moments.fm_fim_relax.relax), `safe_load`'s `torch.jit.load`
# monkeypatch, and `MACEEmbedder.embed`'s hook on the shared backbone. Each of
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
    trained that way), but ``torch.set_default_dtype`` is process-wide.
    ``MagneticMACECalculator`` sets float64 around its forward pass only, with
    a save and restore that takes no lock: it collects results into tensors
    built at whatever the default is outside that scope (float32 energies,
    otherwise), and two threads running mace at once can restore each other's
    value. Wrap mace's construction and calls in this block, so they run in
    float64 and one at a time under ``_mace_lock``.

    It does not protect other models. ``_mace_lock`` serializes mace callers
    only, so while one thread is inside this block every other thread in the
    process sees the float64 default too. Each float32 model fixes its own
    dtype when it is built instead (``load_cgcnn`` and the ``is_metal`` and
    ``is_magnetic`` predictors) -- see goldilocks-ml#110.

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


def _atoms_to_batch(atoms: Any, raw_model: Any, device: str) -> dict[str, Any]:
    """Build the backbone's input for ``atoms``, moments under ``MAGMOM_KEY``.

    The same batch ``MagneticMACECalculator`` builds, as a dict of float64
    tensors on ``device``.
    """
    from mace import data as mace_data
    from mace.tools import torch_geometric, utils

    # The calculator's defaults, so the batch matches the one it builds.
    heads = list(raw_model.heads)
    head = (
        heads[0]
        if len(heads) == 1
        else next(name for name in heads if name.lower() == "default")
    )
    keyspec = mace_data.KeySpecification(
        info_keys={"total_spin": "spin", "total_charge": "charge"},
        arrays_keys={"charges": "Qs", "magmom": MAGMOM_KEY},
    )
    z_table = utils.AtomicNumberTable([int(z) for z in raw_model.atomic_numbers])

    with _default_dtype_float64():
        config = mace_data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=head
        )
        data = mace_data.AtomicData.from_config(
            config, z_table=z_table, cutoff=float(raw_model.r_max), heads=heads
        )
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[data], batch_size=1, shuffle=False, drop_last=False
        )
        return next(iter(loader)).to(device).to_dict()


def _zero_moment_forward(atoms: Any, raw_model: Any, device: str) -> dict[str, Any]:
    """Run the backbone forward once on ``atoms`` at zero moment, without gradients.

    Asks for no forces, stress or magnetic forces, so no autograd graph is
    built. Returns the backbone's output dict; ``"energy"`` is in eV.
    """
    import torch

    atoms = atoms.copy()
    atoms.arrays[MAGMOM_KEY] = np.zeros((len(atoms), 3), dtype=float)

    with _default_dtype_float64(), torch.no_grad():
        return raw_model(
            _atoms_to_batch(atoms, raw_model, device),
            training=False,
            compute_force=False,
            compute_stress=False,
        )


class MACEEmbedder:
    """Pool one zero-moment forward pass into a structure vector."""

    def __init__(self, raw_model: Any, device: str = "cpu") -> None:
        self.raw_model = raw_model
        self.device = device

    def embed(self, atoms: Any) -> np.ndarray:
        """Return the pooled embedding for one ASE ``Atoms``.

        ``load_embedder`` hands the same cached instance, and so the same
        ``raw_model.products[-1]``, to every caller. The hook is registered,
        fired and removed under ``_mace_lock``, and writes into a buffer local
        to this call, so concurrent callers never see each other's features.
        """
        import torch

        captured: dict[str, Any] = {}

        def hook(module: Any, inputs: Any, output: Any) -> None:
            features = output[0] if isinstance(output, tuple) else output
            captured["features"] = features.detach().cpu()

        with _mace_lock:
            handle = self.raw_model.products[-1].register_forward_hook(hook)
            try:
                _zero_moment_forward(atoms, self.raw_model, self.device)
            finally:
                handle.remove()
        features = captured.get("features")

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

    torch.serialization.add_safe_globals([slice])
    return MACEEmbedder(safe_load(Path(checkpoint), device), device)


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
