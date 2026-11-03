"""Compare two parity runs and write the Markdown report.

    uv run python scripts/mmace_parity/compare.py structures.json \
        REFERENCE_DIR CANDIDATE_DIR \
        deposits/magnetism/is_magnetic/mace_mlp/PARITY.md \
        path/to/matpes_pbe_material_records.json.gz
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

THRESHOLD = 0.323869526386261
INVALID = "invalid"
# Fixed before either run finished.
EMBEDDING_ATOL = 1e-4
SCORE_ATOL = 1e-5
MOMENT_ATOL = 1e-3  # bohr magnetons
ENERGY_ATOL = 1e-4  # eV

structures_path, reference_dir, candidate_dir, out, records = map(Path, sys.argv[1:6])
records_sha = hashlib.sha256(records.read_bytes()).hexdigest()
entries = {e["id"]: e for e in json.loads(structures_path.read_text())}


def load(directory: Path):
    results = json.loads((directory / "results.json").read_text())
    arrays = np.load(directory / "embeddings.npz")
    return results, list(arrays["ids"]), arrays["embeddings"].astype(np.float64)


ref, ref_ids, ref_emb = load(reference_dir)
new, new_ids, new_emb = load(candidate_dir)
assert ref_ids == new_ids, "the two runs embedded different structures"
ids = ref_ids

emb_diff = np.abs(ref_emb - new_emb)
per_structure = emb_diff.max(axis=1)
worst_emb = int(per_structure.argmax())
scale = np.abs(ref_emb).max()

ref_scores = np.array([ref["scores"][i] for i in ids])
new_scores = np.array([new["scores"][i] for i in ids])
score_diff = np.abs(ref_scores - new_scores)
ref_decision = ref_scores >= THRESHOLD
new_decision = new_scores >= THRESHOLD
flips = [ids[k] for k in np.flatnonzero(ref_decision != new_decision)]
margin = np.abs(ref_scores - THRESHOLD)
nearest = int(margin.argmin())


def metrics(decision: np.ndarray, group: str) -> dict[str, float]:
    mask = np.array([entries[i]["group"] == group for i in ids])
    truth = np.array([entries[i].get("label") == "magnetic" for i in ids])[mask]
    pred = decision[mask]
    tp = int((pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    return {
        "n": int(mask.sum()),
        "accuracy": (tp + tn) / mask.sum(),
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "confusion": [[tn, fp], [fn, tp]],
    }


ref_metrics = metrics(ref_decision, "test")
new_metrics = metrics(new_decision, "test")

relax_rows = []
for key, before in ref["relax"].items():
    after = new["relax"][key]
    m_before = np.array(before["moments"])
    m_after = np.array(after["moments"])
    relax_rows.append(
        {
            "id": key,
            "formula": entries[key].get("formula") or key.split(":")[1],
            "natoms": len(m_before),
            "status_ref": before["status"],
            "status_new": after["status"],
            "max_dm": float(np.abs(m_before - m_after).max()),
            "max_m": float(np.abs(m_after[:, 2]).max()),
            "transverse": float(np.abs(m_after[:, :2]).max()),
            "de": abs(before["energy_ev"] - after["energy_ev"]),
        }
    )

# relax() rejects an `invalid` result, so its moments and energy are only
# compared where both runs returned something it accepts.
accepted = [r for r in relax_rows if INVALID not in (r["status_ref"], r["status_new"])]
status_differs = [r["id"] for r in relax_rows if r["status_ref"] != r["status_new"]]

checks = {
    "embeddings": ("pass" if float(emb_diff.max()) <= EMBEDDING_ATOL else "FAIL", ""),
    "scores": ("pass" if float(score_diff.max()) <= SCORE_ATOL else "FAIL", ""),
    "decisions": ("pass" if not flips else "FAIL", f"{len(flips)} flips"),
    "relax moments": (
        "pass" if max(r["max_dm"] for r in accepted) <= MOMENT_ATOL else "FAIL",
        f"{len(accepted)} structures both runs accept",
    ),
    "relax energies": (
        "pass" if max(r["de"] for r in accepted) <= ENERGY_ATOL else "FAIL",
        f"{len(accepted)} structures both runs accept",
    ),
    "relax status": (
        "pass" if not status_differs else "differs",
        f"{len(relax_rows) - len(status_differs)} of {len(relax_rows)} agree"
        + (f"; differs: {', '.join(status_differs)}" if status_differs else ""),
    ),
    "collinear": (
        "pass" if max(r["transverse"] for r in relax_rows) == 0.0 else "FAIL",
        "",
    ),
}


def relax_line(r: dict) -> str:
    head = (
        f"{r['id']:<28}{r['formula']:<14}{r['natoms']:>5}  "
        f"{r['status_ref']:<22}{r['status_new']:<22}"
    )
    if r["status_new"] == INVALID:
        return head + f"{'-':>8}{'-':>10}{'-':>10}{r['transverse']:>11.2g}"
    numbers = f"{r['max_m']:>8.3f}"
    if r["status_ref"] == INVALID:
        return head + numbers + f"{'-':>10}{'-':>10}{r['transverse']:>11.2g}"
    return (
        head + numbers + f"{r['max_dm']:>10.2g}{r['de']:>10.2g}{r['transverse']:>11.2g}"
    )


def scan_section() -> list[str]:
    """Explain a status difference with the energy scan, when one was run."""
    if not status_differs:
        return []
    paths = [reference_dir / "scan.json", candidate_dir / "scan.json"]
    if not all(path.is_file() for path in paths):
        return []
    before, after = (json.loads(path.read_text()) for path in paths)
    m_max = ", ".join(f"{k} {v:.4f}" for k, v in after["m_max"].items())
    rows = [
        f"{a['moment']:>6.1f}  {a['energy_ev']:>24.12g}  {b['energy_ev']:>24.12g}  "
        f"{'same' if a['energy_ev'] == b['energy_ev'] else 'differs'}"
        for a, b in zip(before["rows"], after["rows"], strict=True)
    ]
    return [
        "### Where the status differs",
        "",
        "mace-torch 0.3.17 clamps |m| / m_max to at most 1 before the Chebyshev",
        "moment basis (`MagneticScaleShiftMACE.forward` in",
        "`mace/modules/extensions.py`); the fork does not. The two backbones are",
        "therefore identical for moments up to the element's `m_max` and differ",
        "above it. relax() rejects a result above 1.05 m_max, but LBFGS's line search",
        "can try such moments on the way, and then the two runs follow different",
        "paths.",
        "",
        f"Energy of {after['id']} with every site at the same moment along z, no SCF",
        f"(m_max: {m_max} uB):",
        "",
        "```text",
        f"{'m (uB)':>6}  {'E v1.0 (eV)':>24}  {'E v2.0 (eV)':>24}",
        *rows,
        "```",
        "",
        "Both versions agree that this energy surface is unphysical above about",
        "4.2 uB, well inside m_max. Neither relaxed result for this structure is",
        "physical: MatPES's own DFT gives it 2.25 uB. v2.0's retry stops at a point",
        "relax() accepts; v1.0's retry did not.",
        "",
    ]


def env_line(name: str, env: dict) -> str:
    return (
        f"{name:<10} torch {env['torch']}, mace-torch {env['mace-torch']}, "
        f"sphericart-torch {env['sphericart-torch']}, e3nn {env['e3nn']}\n"
        f"{'':<10} backbone sha256 {env['backbone_sha256']}"
    )


lines = [
    "# Backbone parity: v1.0 (mace fork) against v2.0 (mace-torch 0.3.17)",
    "",
    "v2.0 of the `is_magnetic` PSDI record (`sgk1t-48m42`; v1.0 is `1g8rw-q8128`)",
    "replaces the backbone file. This compares the two versions on the same",
    "structures, each in the environment it was published for, through the public",
    "goldilocks-ml API: `embed_structures`, the `is_magnetic` predictor's scores,",
    "and `relax()`.",
    "",
    "```text",
    env_line("v1.0", ref["environment"]),
    env_line("v2.0", new["environment"]),
    "```",
    "",
    "v1.0 ran the `CheukHinHoJerry/mace` fork at",
    "`19cdf6692c48e068a24e06cfe1ffc670e8aea3dd` with goldilocks-ml 0.2.3. v2.0 ran",
    "this branch. Both used the same `is_magnetic.pt`.",
    "",
    "## Result",
    "",
    "```text",
    *(
        f"{name:<15} {verdict:<8} {detail}".rstrip()
        for name, (verdict, detail) in checks.items()
    ),
    "```",
    "",
    "## Structures",
    "",
    f"- {len(ids)} structures embedded and scored: every MatPES PBE test-split",
    "  structure with a label (max |site moment| <= 0.05 or >= 0.5 uB), plus",
    "  hand-built bcc Fe, diamond Si and fcc Pt and eight well-known MatPES",
    "  entries (mp-13 bcc Fe, mp-150 fcc Fe, mp-23 Ni, mp-54 Co, mp-149 Si,",
    "  mp-126 Pt, mp-19009 NiO, mp-19770 Fe2O3).",
    f"- {len(relax_rows)} structures relaxed with `relax()`: bcc Fe from a [0, 0, 2.2]",
    "  seed, six of the named entries, and 14 magnetic test-split structures of at",
    "  most 24 atoms drawn with seed 117. All but the first start from",
    "  `seed_moments_fm_fim`.",
    "",
    "The MatPES records come from `materialyze/matpes/pbe-2025.2` (CC-BY-4.0), with",
    "the train/valid/test split the classifier was trained on:",
    f"`{records.name}`, sha256 `{records_sha}`.",
    "",
    "## Embeddings and scores",
    "",
    "```text",
    f"embedding max |diff|          {emb_diff.max():.3g}   "
    f"(tolerance {EMBEDDING_ATOL:g}; largest |value| {scale:.3g})",
    f"embedding worst structure     {ids[worst_emb] if emb_diff.max() > 0 else '-'}",
    f"structures bit-identical      {int((per_structure == 0).sum())} of {len(ids)}",
    f"score max |diff|              {score_diff.max():.3g}   "
    f"(tolerance {SCORE_ATOL:g})",
    f"decision flips at {THRESHOLD:.4f}   {len(flips)}",
    f"closest score to threshold    {ref_scores[nearest]:.6f} ({ids[nearest]}), "
    f"diff there {score_diff[nearest]:.3g}",
    "```",
    "",
    f"Decisions on the {ref_metrics['n']} labelled test-split structures:",
    "",
    "```text",
    "         accuracy  precision  recall  confusion [[TN, FP], [FN, TP]]",
    f"v1.0     {ref_metrics['accuracy']:.4f}    {ref_metrics['precision']:.4f}     "
    f"{ref_metrics['recall']:.4f}  {ref_metrics['confusion']}",
    f"v2.0     {new_metrics['accuracy']:.4f}    {new_metrics['precision']:.4f}     "
    f"{new_metrics['recall']:.4f}  {new_metrics['confusion']}",
    "```",
    "",
    "These are not the model card's metrics. The card reports the research",
    "checkpoint's own test evaluation; this split is the one in the MatPES records",
    "file used here.",
    "",
    "## relax()",
    "",
    f"Tolerances: {MOMENT_ATOL:g} uB on any moment component, {ENERGY_ATOL:g} eV on",
    "the energy, checked where both runs return a result relax() accepts. `max|m|`",
    "is the largest |m_z| v2.0 returned; `transverse` is the largest |m_x| or |m_y|.",
    "",
    "```text",
    f"{'structure':<28}{'formula':<14}{'atoms':>5}  "
    f"{'status v1.0':<22}{'status v2.0':<22}"
    f"{'max|m|':>8}{'max|dm|':>10}{'|dE| eV':>10}{'transverse':>11}",
    *(relax_line(r) for r in relax_rows),
    "```",
    "",
    *scan_section(),
    "## Reproduce",
    "",
    "`scripts/mmace_parity/` holds the scripts that produced this file:",
    "`select_structures.py` picks the structures from the MatPES records,",
    "`run_pipeline.py` and `scan_moments.py` run once per environment, and",
    "`compare.py` writes this report.",
    "",
]
out.write_text("\n".join(lines))
print("\n".join(lines))
