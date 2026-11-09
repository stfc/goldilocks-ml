"""Measure peak memory and time of the mMACE zero-moment forward pass.

Runs the ``is_magnetic`` embedding (``embed_structures``) and ``relax()``'s
zero-moment energy (``_zero_moment_energy``) on bcc Fe supercells, each in a
fresh process, and prints one row per run: peak RSS after loading the
backbone, peak RSS after the call, and the time of a first and a second call
on the same structure.

It measures whichever goldilocks-ml the interpreter imports, so the same
script compares two versions:

    git worktree add ../goldilocks-ml-0.3.2 v0.3.2
    uv run --project ../goldilocks-ml-0.3.2 python -I \\
        scripts/mmace_forward_benchmark.py BACKBONE
    uv run python -I scripts/mmace_forward_benchmark.py BACKBONE
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

TARGETS = ("embed", "zero_moment_energy")


def _peak_rss_gb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # bytes on macOS, KiB on Linux
    return peak / 1e9 if sys.platform == "darwin" else peak * 1024 / 1e9


def _child(target: str, atoms: int, backbone: Path) -> dict:
    import torch
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    import goldilocks_ml
    from goldilocks_ml.models.magnetism._mace_backbone import (
        embed_structures,
        load_embedder,
    )
    from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
        _zero_moment_energy,
    )

    repeat = round((atoms / 2) ** (1 / 3))
    if 2 * repeat**3 != atoms:
        raise SystemExit(f"{atoms} is not a bcc supercell size (2 n^3)")
    structure = Structure(
        Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    ) * (repeat, repeat, repeat)
    embedder = load_embedder(backbone)
    loaded_gb = _peak_rss_gb()

    def call() -> None:
        if target == "embed":
            embed_structures([structure], checkpoint=backbone)
        else:
            _zero_moment_energy(
                AseAtomsAdaptor.get_atoms(structure),
                raw_model=embedder.raw_model,
                device="cpu",
            )

    seconds = []
    for _ in range(2):
        started = time.perf_counter()
        call()
        seconds.append(time.perf_counter() - started)
    return {
        "goldilocks_ml": goldilocks_ml.__version__,
        "torch": torch.__version__,
        "platform": sys.platform,
        "target": target,
        "atoms": atoms,
        "loaded_peak_rss_gb": round(loaded_gb, 3),
        "peak_rss_gb": round(_peak_rss_gb(), 3),
        "first_call_s": round(seconds[0], 3),
        "second_call_s": round(seconds[1], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("backbone", type=Path)
    parser.add_argument("--atoms", type=int, nargs="+", default=[54, 128])
    parser.add_argument("--target", choices=TARGETS, nargs="+", default=TARGETS)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    backbone = args.backbone.resolve()

    if args.child:
        (target,), (atoms,) = args.target, args.atoms
        print(json.dumps(_child(target, atoms, backbone)))
        return
    for target in args.target:
        for atoms in args.atoms:
            command = [sys.executable, "-I", __file__, str(backbone), "--child"]
            command += ["--target", target, "--atoms", str(atoms)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                sys.exit(result.stderr)
            print(result.stdout.strip().splitlines()[-1], flush=True)


if __name__ == "__main__":
    main()
