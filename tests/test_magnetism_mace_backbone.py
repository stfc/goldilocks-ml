"""Tests for the shared mace-backbone loading machinery.

Deliberately mace-free: importing ``_mace_backbone`` never touches
mace/e3nn/sphericart, only calling its loading functions does (see the
module's own docstring), so this exercises ``_default_dtype_float64``
directly without the manually-installed stack.
"""

from __future__ import annotations

import threading

import torch

from goldilocks_ml.models.magnetism._mace_backbone import _default_dtype_float64


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
