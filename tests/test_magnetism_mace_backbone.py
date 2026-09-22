"""Tests for the shared mace-backbone loading machinery.

Deliberately mace-free: importing ``_mace_backbone`` never touches
mace/e3nn/sphericart, only calling its loading functions does (see the
module's own docstring), so this exercises ``_default_dtype_float64``
directly without the manually-installed stack.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
import torch

from goldilocks_ml.models.magnetism._mace_backbone import (
    MACEEmbedder,
    _default_dtype_float64,
    _mace_lock,
    safe_load,
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
    """Stands in for ``raw_model.products[-1]``: a real ``nn.Module`` forward
    hook registered on the object every ``embed()`` call shares, since
    ``load_embedder`` caches one ``MACEEmbedder`` per checkpoint."""

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
    """Stands in for an ASE ``Atoms`` + calculator pair, with a controllable
    delay so two ``embed()`` calls can be forced to overlap in time."""

    def __init__(self, layer: _FakeProductLayer, value: float, delay: float) -> None:
        self.layer = layer
        self.value = value
        self.delay = delay
        self.arrays: dict[str, object] = {}
        self.calc: object = None

    def __len__(self) -> int:
        return 2

    def copy(self) -> _FakeAtoms:
        return self

    def get_potential_energy(self) -> float:
        time.sleep(self.delay)
        self.layer.fire((torch.full((2, 8), self.value),))
        return 0.0


def test_embed_holds_the_lock_for_its_whole_body() -> None:
    """Regression test for a race far worse than a crash: two ``embed()``
    calls sharing the same cached ``MACEEmbedder`` instance (as
    ``load_embedder``'s cache does for every real caller) could silently
    return *each other's* embeddings with no error at all, if the shared
    hook/buffer were only locked around the inference step rather than the
    whole method -- one thread's stale, not-yet-removed hook could fire
    during another thread's forward pass and overwrite the shared buffer
    before the first thread read it back. A timing-based repro of that exact
    interleaving is inherently flaky (the vulnerable window is a few
    Python statements wide, not something a ``time.sleep`` mismatch reliably
    hits); what actually matters, and what this asserts directly, is that
    ``embed()`` cannot make *any* progress -- not even clearing the shared
    buffer -- until it holds ``_mace_lock``, by holding the lock externally
    and confirming a concurrent call blocks before touching shared state.
    """
    layer = _FakeProductLayer()
    raw_model = type("RawModel", (), {"products": [None, layer]})()

    class FakeCalculator:
        def reset(self) -> None:
            pass

    embedder = MACEEmbedder(raw_model, FakeCalculator())
    embedder._buffer["sentinel"] = "untouched"

    with _mace_lock:
        thread = threading.Thread(
            target=embedder.embed, args=(_FakeAtoms(layer, 1.0, 0.0),)
        )
        thread.start()
        thread.join(timeout=0.2)
        still_blocked = thread.is_alive()
        buffer_untouched = embedder._buffer.get("sentinel") == "untouched"

    thread.join(timeout=1.0)
    assert still_blocked, "embed() made progress without holding _mace_lock"
    assert buffer_untouched


def test_embed_produces_correct_results_under_concurrent_load() -> None:
    """Sanity check that the lock above doesn't just block, it still lets
    concurrent callers each get their own correct result once serialized."""
    layer = _FakeProductLayer()
    raw_model = type("RawModel", (), {"products": [None, layer]})()

    class FakeCalculator:
        def reset(self) -> None:
            pass

    embedder = MACEEmbedder(raw_model, FakeCalculator())
    results: dict[str, object] = {}
    started = threading.Barrier(2)

    def run(name: str, value: float, delay: float) -> None:
        started.wait()
        results[name] = embedder.embed(_FakeAtoms(layer, value, delay))

    a = threading.Thread(target=run, args=("A", 1.0, 0.2))
    b = threading.Thread(target=run, args=("B", 2.0, 0.05))
    a.start()
    b.start()
    a.join()
    b.join()

    assert results["A"][0] == pytest.approx(1.0)  # type: ignore[index]
    assert results["B"][0] == pytest.approx(2.0)  # type: ignore[index]
