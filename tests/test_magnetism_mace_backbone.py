"""Tests for the shared mace-backbone loading machinery.

The lock tests use fakes. The forward-pass tests run on a small magnetic MACE
with random weights, built from the released backbone's interaction blocks,
so they need no checkpoint; they also run on the released backbone when it is
present locally.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from conftest import tiny_magnetic_backbone
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from goldilocks_ml.models.magnetism import _mace_backbone
from goldilocks_ml.models.magnetism._mace_backbone import (
    MAGMOM_KEY,
    MACEEmbedder,
    _default_dtype_float64,
    _mace_lock,
    _zero_moment_forward,
    safe_load,
)
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
    _zero_moment_energy,
)

RELEASED_BACKBONE = Path(
    "local_data/artifacts/sgk1t-48m42/mmace-matpes-pbe-pretrained.model"
)


def test_default_dtype_float64_serializes_concurrent_callers() -> None:
    """Regression test for a race in goldilocks-ml#98's fix.

    The context manager is a plain save/restore of a process-wide global
    with no lock of its own: without one, two threads racing through it can
    interleave so that thread A restores the dtype *while thread B is still
    computing under it* (corrupting B's computation mid-flight), and B then
    restores the *wrong* "previous" value on exit, leaving the global stuck
    at float64 permanently -- confirmed empirically before adding the lock,
    with the exact barrier-forced interleaving below. FastAPI's threadpool
    (goldilocks-core's real deployment shape) makes this a live scenario,
    not a hypothetical one.
    """
    assert torch.get_default_dtype() == torch.float32
    observed_mid_computation: list[torch.dtype] = []
    entered = threading.Barrier(2)
    may_finish = threading.Barrier(2)

    def thread_a() -> None:
        with _default_dtype_float64():
            entered.wait()  # let B enter while A is still inside
            # A exits (and restores) well before B does.
        may_finish.wait()

    def thread_b() -> None:
        entered.wait()  # only enter once A is confirmed inside
        with _default_dtype_float64():
            # By the time B checks, A has already exited and restored --
            # the buggy version leaked that restore into B's computation.
            may_finish.wait()
            observed_mid_computation.append(torch.get_default_dtype())

    a = threading.Thread(target=thread_a)
    b = threading.Thread(target=thread_b)
    a.start()
    b.start()
    a.join()
    b.join()

    assert observed_mid_computation == [torch.float64]
    assert torch.get_default_dtype() == torch.float32


def test_safe_load_serializes_the_jit_load_monkeypatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the same race applied to ``safe_load``.

    Without a lock, two threads with different ``device`` arguments can
    interleave their own save/restore of ``torch.jit.load`` (also a
    process-wide global with no scoping of its own): whichever restores
    last writes back the *other* thread's monkeypatch instead of the real
    original, leaving ``torch.jit.load`` permanently stuck pointing at the
    wrong device -- confirmed empirically before adding the lock, with two
    threads and slow-but-different-duration fake loads.
    """
    real_jit_load = torch.jit.load

    def slow_fake_torch_load(
        path: object, map_location: object = None, **_: object
    ) -> str:
        time.sleep(0.2 if map_location == "device-A" else 0.05)
        return f"result-for-{map_location}"

    monkeypatch.setattr(torch, "load", slow_fake_torch_load)

    results: dict[str, str] = {}
    entered = threading.Barrier(2)

    def run(name: str, device: str) -> None:
        entered.wait()
        results[name] = safe_load(Path("fake-path"), device=device)

    a = threading.Thread(target=run, args=("A", "device-A"))
    b = threading.Thread(target=run, args=("B", "device-B"))
    a.start()
    b.start()
    a.join()
    b.join()

    assert results == {"A": "result-for-device-A", "B": "result-for-device-B"}
    assert torch.jit.load is real_jit_load


class _FakeProductLayer:
    """Stands in for ``raw_model.products[-1]``: the module every ``embed()``
    call hooks, shared because ``load_embedder`` caches one ``MACEEmbedder``
    per checkpoint."""

    def __init__(self) -> None:
        self._hooks: list[list[object]] = []

    def register_forward_hook(self, fn: object) -> object:
        entry = [fn]
        self._hooks.append(entry)

        class Handle:
            def remove(_self) -> None:
                if entry in self._hooks:
                    self._hooks.remove(entry)

        return Handle()

    def fire(self, output: object) -> None:
        for entry in list(self._hooks):
            entry[0](None, None, output)  # type: ignore[operator]


class _FakeAtoms:
    """Stands in for an ASE ``Atoms``: the value the fake forward pass emits,
    and how long it takes, so two ``embed()`` calls can be forced to overlap."""

    def __init__(self, value: float, delay: float) -> None:
        self.value = value
        self.delay = delay


def _fake_forward(layer: _FakeProductLayer, calls: list[_FakeAtoms]) -> Any:
    def forward(atoms: _FakeAtoms, raw_model: object, device: str) -> dict:
        calls.append(atoms)
        time.sleep(atoms.delay)
        layer.fire((torch.full((2, 8), atoms.value),))
        return {}

    return forward


def test_embed_holds_the_lock_for_its_whole_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for goldilocks-ml#100: two ``embed()`` calls sharing
    the cached ``MACEEmbedder`` could silently return each other's
    embeddings when only the forward pass was locked, because one thread's
    not-yet-removed hook could fire during the other's forward pass. A
    timing-based repro is flaky; this asserts directly that ``embed()``
    neither registers its hook nor runs the forward pass until it holds
    ``_mace_lock``.
    """
    layer = _FakeProductLayer()
    raw_model = type("RawModel", (), {"products": [None, layer]})()
    calls: list[_FakeAtoms] = []
    monkeypatch.setattr(
        _mace_backbone, "_zero_moment_forward", _fake_forward(layer, calls)
    )
    embedder = MACEEmbedder(raw_model)

    with _mace_lock:
        thread = threading.Thread(target=embedder.embed, args=(_FakeAtoms(1.0, 0.0),))
        thread.start()
        thread.join(timeout=0.2)
        still_blocked = thread.is_alive()
        untouched = not layer._hooks and not calls

    thread.join(timeout=1.0)
    assert still_blocked, "embed() made progress without holding _mace_lock"
    assert untouched
    assert len(calls) == 1
    assert not layer._hooks, "embed() left its hook on the shared layer"


def test_embed_produces_correct_results_under_concurrent_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent callers of one cached ``MACEEmbedder`` each get their own
    features, the slower one included."""
    layer = _FakeProductLayer()
    raw_model = type("RawModel", (), {"products": [None, layer]})()
    monkeypatch.setattr(
        _mace_backbone, "_zero_moment_forward", _fake_forward(layer, [])
    )
    embedder = MACEEmbedder(raw_model)
    results: dict[str, object] = {}
    started = threading.Barrier(2)

    def run(name: str, value: float, delay: float) -> None:
        started.wait()
        results[name] = embedder.embed(_FakeAtoms(value, delay))

    a = threading.Thread(target=run, args=("A", 1.0, 0.2))
    b = threading.Thread(target=run, args=("B", 2.0, 0.05))
    a.start()
    b.start()
    a.join()
    b.join()

    assert results["A"][0] == pytest.approx(1.0)  # type: ignore[index]
    assert results["B"][0] == pytest.approx(2.0)  # type: ignore[index]


@pytest.fixture(scope="module", params=["tiny", "released"])
def backbone(request: pytest.FixtureRequest) -> torch.nn.Module:
    if request.param == "tiny":
        return tiny_magnetic_backbone()
    if not RELEASED_BACKBONE.is_file():
        pytest.skip("the mMACE backbone checkpoint is not present locally")
    torch.serialization.add_safe_globals([slice])
    return safe_load(RELEASED_BACKBONE)


_BCC_FE = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
FIXTURES = {
    "bcc-Fe": _BCC_FE,
    "bcc-Fe-54-atoms": _BCC_FE * (3, 3, 3),
    "rocksalt-NaCl-primitive": Structure(
        Lattice.from_parameters(3.99, 3.99, 3.99, 60, 60, 60),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    ),
    "hcp-Co": Structure(
        Lattice.hexagonal(2.507, 4.07),
        ["Co", "Co"],
        [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]],
    ),
    "triclinic-NiO": Structure(
        Lattice.from_parameters(4.1, 4.4, 5.3, 81.0, 97.0, 104.0),
        ["Ni", "Ni", "O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.1], [0.27, 0.31, 0.44], [0.71, 0.64, 0.58]],
    ),
}


def _calculator_path(raw_model: Any, atoms: Any) -> tuple[np.ndarray, float]:
    """goldilocks-ml 0.3.2's embedding and zero-moment energy: the same hook
    and pooling, with the forward pass run by ``MagneticMACECalculator``."""
    from mace.calculators.mace import MagneticMACECalculator
    from mace.modules import MagneticSCFMACE

    one_step_model = MagneticSCFMACE(
        raw_model, n_scf_step=1, use_scf=False, scf_tol=1e-4, scf_logging=False
    )
    with _default_dtype_float64():
        calculator = MagneticMACECalculator(
            models=[one_step_model],
            device="cpu",
            default_dtype="float64",
            magmom_key=MAGMOM_KEY,
        )
    atoms = atoms.copy()
    atoms.arrays[MAGMOM_KEY] = np.zeros((len(atoms), 3))
    captured: dict[str, torch.Tensor] = {}

    def hook(module: Any, inputs: Any, output: Any) -> None:
        features = output[0] if isinstance(output, tuple) else output
        captured["features"] = features.detach().cpu()

    handle = raw_model.products[-1].register_forward_hook(hook)
    try:
        with _default_dtype_float64():
            atoms.calc = calculator
            energy = float(atoms.get_potential_energy())
    finally:
        handle.remove()
    features = captured["features"]
    pooled = torch.cat(
        [
            features.mean(dim=0),
            features.max(dim=0).values,
            features.std(dim=0, unbiased=False),
        ]
    )
    return pooled.numpy().astype(np.float32), energy


@pytest.mark.parametrize("name", list(FIXTURES))
def test_the_forward_pass_matches_the_calculator_path_bit_for_bit(
    backbone: torch.nn.Module, name: str
) -> None:
    """goldilocks-ml#140: the embedding and ``relax()``'s zero-moment energy
    no longer go through ``MagneticMACECalculator``, which also computed
    forces and stress. Both must be unchanged, to the last bit."""
    atoms = AseAtomsAdaptor.get_atoms(FIXTURES[name])
    expected_embedding, expected_energy = _calculator_path(backbone, atoms)
    assert np.any(expected_embedding != 0)

    embedding = MACEEmbedder(backbone).embed(atoms)
    energy = _zero_moment_energy(atoms, raw_model=backbone, device="cpu")

    assert embedding.dtype == np.float32
    assert np.array_equal(embedding, expected_embedding)
    assert energy == expected_energy
    assert torch.get_default_dtype() == torch.float32


def test_the_forward_pass_builds_no_autograd_graph() -> None:
    """The embedding and the zero-moment energy run the backbone with
    gradients off, and ask it for no forces, stress or magnetic forces."""
    raw_model = tiny_magnetic_backbone()
    atoms = AseAtomsAdaptor.get_atoms(FIXTURES["bcc-Fe"])
    grad_enabled: list[bool] = []
    handle = raw_model.products[-1].register_forward_hook(
        lambda module, inputs, output: grad_enabled.append(torch.is_grad_enabled())
    )
    try:
        output = _zero_moment_forward(atoms, raw_model, "cpu")
        MACEEmbedder(raw_model).embed(atoms)
        _zero_moment_energy(atoms, raw_model=raw_model, device="cpu")
    finally:
        handle.remove()

    assert grad_enabled == [False, False, False]
    assert output["energy"].grad_fn is None
    assert output["forces"] is None
    assert output["stress"] is None
    assert output["magforces"] is None
    assert torch.is_grad_enabled()
