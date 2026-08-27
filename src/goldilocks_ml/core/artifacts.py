"""Resolve and verify released model artifacts a feature contract depends on."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from goldilocks_ml.core.hashing import sha256_file
from goldilocks_ml.core.protocol import ArtifactDependency

ENVIRONMENT_VARIABLE = "GOLDILOCKS_ARTIFACTS"
DEFAULT_DIRECTORY = Path("local_data/artifacts")
RECORD_URL = "https://data-collections.psdi.ac.uk/records"


def artifact_directory(override: Path | None = None) -> Path:
    """Return where released artifacts are cached locally."""
    if override is not None:
        return override.resolve()
    from_environment = os.environ.get(ENVIRONMENT_VARIABLE)
    if from_environment:
        return Path(from_environment).resolve()
    return DEFAULT_DIRECTORY.resolve()


def resolve(
    dependencies: Sequence[ArtifactDependency], directory: Path
) -> dict[str, Path]:
    """Return verified local paths for every pinned artifact dependency."""
    resolved: dict[str, Path] = {}
    for dependency in dependencies:
        path = directory / dependency.record_id / dependency.file
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. The {dependency.name} feature dependency is "
                f"file {dependency.file} from PSDI record {dependency.record_id} "
                f"({RECORD_URL}/{dependency.record_id}). Download it to that path."
            )
        digest = sha256_file(path)
        if digest != dependency.sha256:
            raise ValueError(
                f"{path} SHA-256 is {digest}; the protocol pins {dependency.sha256}. "
                "The features this produces would not match the protocol."
            )
        resolved[dependency.name] = path
    return resolved
