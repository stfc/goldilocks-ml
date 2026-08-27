"""A feature contract that reads precomputed model inputs from the snapshot.

Use this when the features are already numbers in ``features.csv``. Contracts
that derive features from structures do their own work instead; see
``goldilocks_ml.features`` for those.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping
from pathlib import Path

from goldilocks_ml.core.protocol import TrainingProtocol
from goldilocks_ml.core.registry import FeatureMatrix, register_feature_contract
from goldilocks_ml.core.snapshot import Snapshot

SCHEMA = "tabular"


def _number(value: str, column: str, sample_id: str) -> float:
    if not value.strip():
        raise ValueError(f"{sample_id} has an empty {column}")
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"{sample_id} has a non-numeric {column}: {value!r}"
        ) from error
    if not math.isfinite(number):
        raise ValueError(f"{sample_id} has a non-finite {column}: {value!r}")
    return number


def build(
    protocol: TrainingProtocol,
    snapshot: Snapshot,
    artifacts: Mapping[str, Path],
) -> FeatureMatrix:
    """Read `features.csv` and select the columns the protocol asked for."""
    if snapshot.features_file is None:
        raise ValueError(
            "the tabular feature contract needs a features file; the snapshot "
            "manifest declares none"
        )
    unknown = sorted(set(protocol.features.parameters) - {"columns"})
    if unknown:
        raise ValueError(f"unknown tabular feature parameter(s): {', '.join(unknown)}")

    path = snapshot.directory / snapshot.features_file
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        header = list(reader.fieldnames)
        table = {row[header[0]].strip(): row for row in reader}

    id_column, *available = header
    if len(available) != len(set(available)):
        raise ValueError(f"{path} has duplicate column names")

    requested = protocol.features.parameters.get("columns")
    if requested is None:
        columns = tuple(available)
    else:
        if not isinstance(requested, list) or any(
            not isinstance(column, str) or not column for column in requested
        ):
            raise ValueError(
                "features.parameters.columns must be an array of non-empty strings"
            )
        columns = tuple(requested)
        if len(columns) != len(set(columns)):
            raise ValueError("features.parameters.columns must be unique")
        missing = [column for column in columns if column not in available]
        if missing:
            raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")
    if not columns:
        raise ValueError(f"{path} provides no feature columns beside {id_column}")

    rows: dict[str, tuple[float, ...]] = {}
    for sample_id in snapshot.sample_ids:
        row = table.get(sample_id)
        if row is None:
            raise ValueError(f"{path} has no row for {sample_id}")
        rows[sample_id] = tuple(
            _number(row[column], column, sample_id) for column in columns
        )
    return FeatureMatrix(columns=columns, rows=rows)


register_feature_contract(SCHEMA, build)
