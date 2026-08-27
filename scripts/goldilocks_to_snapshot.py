"""Convert the published Goldilocks DFT dataset into a training snapshot.

The dataset is PSDI record 75959-bwa52. Unpack `data.tar.gz`, then:

    python scripts/goldilocks_to_snapshot.py \
      --raw local_data/raw/upload_version \
      --output local_data/snapshots/kdist

This is a one-off conversion of our own data, not a general ingest layer.
Anyone else brings data already in the snapshot layout.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goldilocks_ml.cli import seal  # noqa: E402

LEVELS = ("medium", "well", "ultra")
# Two different quantities, both defensible, and they disagree.
#
# "aiida" is the dataset's own Goldilocks_k_distance: the k-point distance a
# user feeds to AiiDA-QuantumESPRESSO, which is what the released model's card
# says it predicts. Every row's value lies inside the interval of distances
# that yields the recorded mesh.
#
# "legacy" is what the historical preprocessing recomputed and actually trained
# on: max(|b_i| / n_i), the tight lower bound implied by the mesh. Reproducing
# the published QRF95 requires this one.
TARGETS = ("aiida", "legacy")
# The search that produced the dataset stopped here, so these rows are
# right-censored: the true distance is at least this large.
CENSORED_AT = 1.0
ELEMENT = re.compile(r"([A-Z][a-z]?)(\d*)")
CELL_LENGTH = "_cell_length_{}"
CELL_ANGLE = "_cell_angle_{}"


def _cell(text: str) -> dict[str, float]:
    cell = {}
    for key in ("a", "b", "c"):
        cell[key] = float(
            re.search(CELL_LENGTH.format(key) + r"\s+([\d.]+)", text).group(1)
        )
    for key in ("alpha", "beta", "gamma"):
        cell[key] = float(
            re.search(CELL_ANGLE.format(key) + r"\s+([\d.]+)", text).group(1)
        )
    return cell


def reciprocal_lengths(text: str) -> tuple[float, float, float]:
    """Return |b_1|, |b_2|, |b_3| in the 2*pi convention pymatgen uses."""
    cell = _cell(text)
    cosines = [math.cos(math.radians(cell[k])) for k in ("alpha", "beta", "gamma")]
    sines = [math.sin(math.radians(cell[k])) for k in ("alpha", "beta", "gamma")]
    ca, cb, cg = cosines
    volume = (
        cell["a"]
        * cell["b"]
        * cell["c"]
        * math.sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg)
    )
    return (
        2 * math.pi * cell["b"] * cell["c"] * sines[0] / volume,
        2 * math.pi * cell["a"] * cell["c"] * sines[1] / volume,
        2 * math.pi * cell["a"] * cell["b"] * sines[2] / volume,
    )


def legacy_distance(cif_text: str, mesh: tuple[int, ...]) -> float:
    """Recompute the target the historical pipeline trained on."""
    return max(b / n for b, n in zip(reciprocal_lengths(cif_text), mesh, strict=True))


def reduced_formula(formula: str) -> str:
    """Return a composition-normalised grouping key.

    `O16Si8` and `O24Si12` are both silica in different cells; grouping by the
    raw string would put them in different splits and leak one into the other.
    """
    counts: Counter[str] = Counter()
    for symbol, digits in ELEMENT.findall(formula):
        if symbol:
            counts[symbol] += int(digits) if digits else 1
    if not counts:
        raise ValueError(f"cannot parse formula {formula!r}")
    values = list(counts.values())
    divisor = math.gcd(*values) if len(values) > 1 else values[0]
    return "".join(f"{symbol}{counts[symbol] // divisor}" for symbol in sorted(counts))


def convert(
    raw: Path, output: Path, *, level: str, target: str, drop_censored: bool
) -> dict[str, int]:
    """Write id_prop.csv and one CIF per sample into an unsealed snapshot."""
    summary = raw / "summary.csv"
    details = raw / "structure_calc_details"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    output.mkdir(parents=True, exist_ok=True)

    target_column = f"{level}_k_distance"
    counts = Counter()
    records: list[tuple[str, str, str]] = []
    for row in csv.DictReader(summary.open(encoding="utf-8")):
        counts["rows"] += 1
        sample_id = row["source_db_id"].strip()
        value = row[target_column].strip()
        if not sample_id or not value:
            counts["no_target"] += 1
            continue
        if drop_censored and float(value) >= CENSORED_AT:
            counts["censored"] += 1
            continue
        source = details / sample_id / f"{sample_id}.cif"
        if not source.is_file():
            counts["no_structure"] += 1
            continue
        if target == "legacy":
            mesh_text = row[f"{level} k-mesh"].strip()
            if not mesh_text:
                counts["no_mesh"] += 1
                continue
            mesh = ast.literal_eval(mesh_text)
            value = f"{legacy_distance(source.read_text(), mesh):.6f}"
        shutil.copyfile(source, output / f"{sample_id}.cif")
        records.append((sample_id, value, reduced_formula(row["Formula"])))
        counts["kept"] += 1

    if not records:
        raise ValueError(f"no usable rows for level {level!r}")
    with (output / "id_prop.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(records)
    counts["groups"] = len({group for _, _, group in records})
    return counts


def main() -> None:
    """Convert the raw dataset and seal the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", choices=LEVELS, default="ultra")
    parser.add_argument("--target", choices=TARGETS, default="aiida")
    parser.add_argument("--record-id", default="goldilocks-kdist")
    parser.add_argument("--version", default="75959-bwa52")
    parser.add_argument(
        "--drop-censored",
        action="store_true",
        help=f"drop rows whose distance hit the search ceiling of {CENSORED_AT}",
    )
    args = parser.parse_args()

    counts = convert(
        args.raw,
        args.output,
        level=args.level,
        target=args.target,
        drop_censored=args.drop_censored,
    )
    result = seal(
        args.output,
        record_id=f"{args.record_id}-{args.level}",
        snapshot_version=f"{args.version}-{args.target}",
        structure_suffix=".cif",
    )
    print(
        f"{counts['kept']} samples in {counts['groups']} composition groups "
        f"from {counts['rows']} rows "
        f"(skipped: no target {counts['no_target']}, "
        f"no structure {counts['no_structure']}, no mesh {counts['no_mesh']}, "
        f"censored {counts['censored']})"
    )
    print(f"manifest SHA-256 {result['manifest_sha256']}")


if __name__ == "__main__":
    main()
