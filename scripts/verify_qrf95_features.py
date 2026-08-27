"""Test the reproduced feature pipeline against the published QRF95 model.

Builds the 483-column feature vector for a sample of structures and feeds it to
the released `QRF95.pkl`. Most of these structures were in that model's training
set, so if the features are right its predictions track the truth tightly; if a
block is wrong, ordered differently, or scaled differently, they will not.

    uv run python scripts/verify_qrf95_features.py --limit 60
"""

from __future__ import annotations

import argparse
import csv
import pickle
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

import goldilocks_ml.models  # noqa: E402, F401  (registers the contracts)
from goldilocks_ml.artifacts import resolve  # noqa: E402
from goldilocks_ml.protocol import load_protocol  # noqa: E402
from goldilocks_ml.registry import get_feature_contract  # noqa: E402
from goldilocks_ml.snapshot import load_snapshot  # noqa: E402

PROTOCOL = Path("src/goldilocks_ml/models/kmesh/qrf95/protocol.toml")
PUBLISHED = Path("local_data/models/kmesh/qrf95/QRF95.pkl")


def main() -> None:
    """Build features for a sample and score them with the published model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=Path("local_data/snapshots/kdist-legacy")
    )
    parser.add_argument("--artifacts", type=Path, default=Path("local_data/artifacts"))
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    protocol = load_protocol(PROTOCOL)
    snapshot = load_snapshot(args.snapshot, protocol)
    truth = {
        row[0]: float(row[1])
        for row in csv.reader((args.snapshot / "id_prop.csv").open(encoding="utf-8"))
    }

    subset = snapshot.samples[: args.limit]
    trimmed = type(snapshot)(
        directory=snapshot.directory,
        record_id=snapshot.record_id,
        snapshot_version=snapshot.snapshot_version,
        manifest_sha256=snapshot.manifest_sha256,
        capabilities=snapshot.capabilities,
        features_file=snapshot.features_file,
        samples=subset,
    )

    artifacts = resolve(protocol.features.depends_on, args.artifacts.resolve())
    started = time.monotonic()
    features = get_feature_contract(protocol.features.schema)(
        protocol, trimmed, artifacts
    )
    elapsed = time.monotonic() - started
    matrix = np.asarray(features.matrix(subset))
    print(
        f"features: {matrix.shape} in {elapsed:.1f}s "
        f"({elapsed / len(subset):.2f}s per structure)"
    )

    model = pickle.loads(PUBLISHED.read_bytes())
    print(f"published model expects {model.n_features_in_} features")
    predictions = np.asarray(model.predict(matrix))
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    low, median, high = predictions[0], predictions[1], predictions[2]

    actual = np.array([truth[sample.sample_id] for sample in subset])
    errors = np.abs(median - actual)
    inside = np.mean((actual >= low) & (actual <= high))
    ordered = np.mean((low <= median) & (median <= high))
    correlation = np.corrcoef(median, actual)[0, 1]

    print(f"quantiles ordered  : {ordered:.1%} of samples")
    print(
        f"median vs truth    : MAE {errors.mean():.4f}, "
        f"median |err| {statistics.median(errors):.4f}, r {correlation:.4f}"
    )
    print(f"90% interval covers: {inside:.1%} of samples")
    print(f"truth  range       : {actual.min():.3f} - {actual.max():.3f}")
    print(f"median range       : {median.min():.3f} - {median.max():.3f}")


if __name__ == "__main__":
    main()
