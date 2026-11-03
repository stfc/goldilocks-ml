"""Embed, score and relax the parity structures in the current environment.

Run it once per environment, each with the backbone published for it.
RECORD_DIR holds ``model.json`` and ``is_magnetic.pt``:

    python scripts/mmace_parity/run_pipeline.py \
        structures.json RECORD_DIR BACKBONE OUT_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from goldilocks_ml.inference import load_model
from goldilocks_ml.models.magnetism._mace_backbone import embed_structures, safe_load
from goldilocks_ml.models.magnetism.is_magnetic.mace_mlp.trainer import class_scores
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.relax import (
    domain_limits,
    relax,
)
from goldilocks_ml.models.magnetism.magnetic_moments.fm_fim_relax.seed_moments import (
    seed_moments_fm_fim,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("structures", type=Path)
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("backbone", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-relax", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads(args.structures.read_text())
    embed_entries = [e for e in entries if e["embed"]][: args.limit]
    relax_entries = [e for e in entries if e.get("relax")]

    environment = {
        name: md.version(name)
        for name in ("torch", "mace-torch", "sphericart-torch", "e3nn", "goldilocks-ml")
    }
    environment["backbone_sha256"] = sha256(args.backbone)

    predictor = load_model(args.record_dir, artifacts={"mace_backbone": args.backbone})

    ids = [e["id"] for e in embed_entries]
    rows = []
    started = time.perf_counter()
    batch = 64
    for start in range(0, len(embed_entries), batch):
        window = [
            Structure.from_dict(e["structure"])
            for e in embed_entries[start : start + batch]
        ]
        rows.append(embed_structures(window, checkpoint=args.backbone))
        print(
            f"embedded {start + len(window)}/{len(embed_entries)} "
            f"in {time.perf_counter() - started:.0f}s",
            flush=True,
        )
    embeddings = np.concatenate(rows) if rows else np.zeros((0, 384), np.float32)
    scores = class_scores(
        predictor.model,
        torch.tensor(embeddings, dtype=torch.float32),
        predictor.scaler_mean,
        predictor.scaler_scale,
    )
    np.savez_compressed(
        args.out_dir / "embeddings.npz", ids=np.array(ids), embeddings=embeddings
    )

    relaxed = {}
    if not args.skip_relax:
        limits = domain_limits(safe_load(args.backbone))
        for entry in relax_entries:
            structure = Structure.from_dict(entry["structure"])
            if entry.get("seed") is not None:
                seed = np.array(entry["seed"], dtype=float)
            else:
                seed = seed_moments_fm_fim(structure, max_moment_by_symbol=limits)
            t0 = time.perf_counter()
            result = relax(structure, seed, checkpoint=args.backbone)
            relaxed[entry["id"]] = {
                "seed": seed.tolist(),
                "moments": result.moments.tolist(),
                "energy_ev": result.energy_ev,
                "status": result.status,
                "reasons": list(result.reasons),
                "seconds": time.perf_counter() - t0,
            }
            seconds = relaxed[entry["id"]]["seconds"]
            print(
                f"relaxed {entry['id']}: {result.status} in {seconds:.1f}s",
                flush=True,
            )

    (args.out_dir / "results.json").write_text(
        json.dumps(
            {
                "environment": environment,
                "embed_seconds": time.perf_counter() - started,
                "scores": dict(zip(ids, (float(s) for s in scores), strict=True)),
                "relax": relaxed,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
