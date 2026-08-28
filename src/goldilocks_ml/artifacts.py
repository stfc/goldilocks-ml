"""Resolve and verify released model artifacts a feature contract depends on."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from goldilocks_ml.hashing import sha256_file
from goldilocks_ml.protocol import ArtifactDependency

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
    dependencies: Sequence[ArtifactDependency],
    directory: Path,
    overrides: Mapping[str, Path] | None = None,
) -> dict[str, Path]:
    """Return verified local paths for every pinned artifact dependency.

    An override supplies the path for one dependency without excusing it from
    verification: a caller that points at its own file still gets the digest
    the protocol pinned checked against it.
    """
    resolved: dict[str, Path] = {}
    supplied = dict(overrides or {})
    for dependency in dependencies:
        path = supplied.get(
            dependency.name, directory / dependency.record_id / dependency.file
        )
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
