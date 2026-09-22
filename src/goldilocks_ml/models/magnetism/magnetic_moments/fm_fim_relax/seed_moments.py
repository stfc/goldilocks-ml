"""Guess an FM/ferrimagnetic initial moment per site, from oxidation states alone.

Ported and merged from two places in
``2-research/2-mace/1-magnetic-mace``: ``ProbeMACEEmbedder._ionic_shell_occupancy``
/ ``maximum_moments`` (``scripts/magnetic_classifier_model.py``), which guesses
oxidation states automatically via pymatgen, and ``initial_moment_for_element``
(``notebooks/01_basic_magnetic_scf.ipynb``), which required a hand-written
per-material shell-occupancy table. This module keeps only the automatic path
-- a hand-curated table does not generalise past the ten CIFs it was written
for.

One deliberate change from the source: ``maximum_moments`` read a loaded
MACE model's ``m_max``/``atomic_numbers`` buffers directly to clamp the guess
inside the backbone's learned domain. Here that clamp is an explicit
parameter (``max_moment_by_symbol``) instead, so this module has no MACE
dependency at all and every function is testable with a plain dict. The
production caller (:mod:`~.relax`) builds that mapping from a loaded
backbone before calling in.

Every moment this produces points along +z, one scalar per chemical element
applied to every site of that element. That is what makes the result FM or
ferrimagnetic by construction, and also what it cannot do: no code here
assigns opposite signs to symmetry-equivalent sites of the same element,
which is what an antiferromagnetic sublattice initialisation needs.
"""

from __future__ import annotations

import queue
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core.structure import Structure

# The fraction of the backbone's learned per-element moment-magnitude domain
# a guess is clamped to. The one-body energy term the backbone learned
# becomes singular near the domain boundary itself, so a guess is kept well
# inside it rather than aimed at the boundary.
DOMAIN_SAFETY_FACTOR = 0.7
# How long an oxidation-state guess may run before it is treated as having
# failed. `Composition.oxi_state_guesses` searches combinatorially and can
# hang on an unusual composition; this bounds it.
OXIDATION_GUESS_TIMEOUT_SECONDS = 2.0


def high_spin_moment(shell: str, electron_count: float) -> float:
    """Return the spin-only high-spin moment for ``d^n`` or ``f^n``."""
    shell = shell.lower()
    capacity = {"d": 10, "f": 14}
    if shell not in capacity:
        raise ValueError(f"shell must be 'd' or 'f', got {shell!r}")
    n = int(electron_count)
    if n != electron_count or not 0 <= n <= capacity[shell]:
        raise ValueError(f"invalid occupancy {shell}^{electron_count}")
    return float(min(n, capacity[shell] - n))


def ionic_shell_occupancy(symbol: str, oxidation: float) -> tuple[str, float] | None:
    """Return the valence ``(shell, occupancy)`` left after removing outer electrons.

    Prefers a partially filled f shell; otherwise falls back to the outermost
    d shell. Returns ``None`` for an element with neither -- there is nothing
    for a magnetic moment to come from.
    """
    from pymatgen.core.periodic_table import Element

    orbitals = Element(symbol).full_electronic_structure
    f_shells = [item for item in orbitals if item[1] == "f" and 0 < item[2] < 14]
    candidates = f_shells or [item for item in orbitals if item[1] == "d"]
    if not candidates:
        return None
    n_principal, shell, neutral_count = candidates[-1]
    capacity = 14 if shell == "f" else 10
    outer_count = sum(electrons for n, _, electrons in orbitals if n > n_principal)
    if oxidation >= 0:
        count = neutral_count - max(float(oxidation) - outer_count, 0.0)
    else:
        count = neutral_count - float(oxidation)
    count = float(np.clip(count, 0.0, capacity))
    return shell, count


@lru_cache(maxsize=256)
def _guess_oxidation_states(
    composition_key: tuple[tuple[str, int], ...],
) -> Mapping[str, float]:
    """Return the most likely oxidation state per element, or ``{}`` if none."""
    from pymatgen.core.composition import Composition

    outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def guess() -> None:
        try:
            result = Composition(dict(composition_key)).oxi_state_guesses(max_sites=-1)
            outcome.put((True, result))
        except Exception as error:  # noqa: BLE001 -- re-raised on the caller's thread below
            outcome.put((False, error))

    # A signal-based timeout (SIGALRM) only works on the main thread of the
    # main interpreter, which this cannot assume -- a web framework's
    # threadpool or an async runtime's worker thread calls in here too.
    # Running the search on its own thread and giving up on *waiting for it*
    # times out from any caller thread; it does not stop the search itself
    # (Python cannot forcibly kill a thread), so an abandoned search keeps
    # running in the background until it finishes on its own. That thread
    # must be a daemon thread, not a `ThreadPoolExecutor` one: the latter
    # registers every worker with `concurrent.futures.thread`'s atexit hook,
    # which unconditionally joins it at interpreter shutdown regardless of
    # `shutdown(wait=False)` -- confirmed empirically, an abandoned
    # `ThreadPoolExecutor` task hangs process exit until it finishes, turning
    # "an unusual composition is slow" into "the process will not exit". A
    # daemon thread carries no such hook and is simply dropped at shutdown.
    threading.Thread(target=guess, daemon=True).start()
    try:
        ok, payload = outcome.get(timeout=OXIDATION_GUESS_TIMEOUT_SECONDS)
    except queue.Empty:
        return {}
    if not ok:
        raise payload  # type: ignore[misc]
    guesses = payload
    return guesses[0] if guesses else {}


def guess_oxidation_states(symbols: Sequence[str]) -> Mapping[str, float]:
    """Return the most likely oxidation state per element in a composition.

    Cached by composition, since a batch of structures sharing chemistries
    (as a snapshot usually does) would otherwise repeat the same combinatorial
    search once per structure. Clear the cache with
    ``guess_oxidation_states.cache_clear`` if that matters for a test.
    """
    counts = Counter(symbols)
    return _guess_oxidation_states(tuple(sorted(counts.items())))


guess_oxidation_states.cache_clear = (  # type: ignore[attr-defined]
    _guess_oxidation_states.cache_clear
)


def seed_moments_fm_fim(
    structure: Structure, *, max_moment_by_symbol: Mapping[str, float]
) -> np.ndarray:
    """Return one FM/ferrimagnetic initial moment per site, an ``(N, 3)`` array.

    Every entry lies along +z. ``max_moment_by_symbol`` names the backbone's
    learned per-element moment-magnitude domain (symbol -> maximum bohr
    magnetons); an element absent from it is left at zero rather than guessed
    past a domain the caller never declared.
    """
    symbols = [str(site.specie.symbol) for site in structure]
    oxidation = guess_oxidation_states(symbols)
    moments = np.zeros((len(structure), 3), dtype=float)
    for index, symbol in enumerate(symbols):
        limit = max_moment_by_symbol.get(symbol)
        if limit is None:
            continue
        occupancy = ionic_shell_occupancy(symbol, oxidation.get(symbol, 0.0))
        if occupancy is None:
            continue
        shell, electrons = occupancy
        capacity = 14 if shell == "f" else 10
        # Not `high_spin_moment(shell, electrons)`: that helper requires an
        # integer occupancy (a hand-specified d^n/f^n), while `electrons` here
        # is a continuous estimate derived from a fractional guessed
        # oxidation state, so the high-spin count is computed directly.
        guess = min(electrons, capacity - electrons)
        moments[index, 2] = min(guess, DOMAIN_SAFETY_FACTOR * limit)
    return moments
