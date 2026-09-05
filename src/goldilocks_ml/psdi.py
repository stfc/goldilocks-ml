"""Validate and upload reproducible model deposits to PSDI Data Collections."""

from __future__ import annotations

import argparse
import json
import stat
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data_collections_api.invenio import InvenioRepository
from data_collections_api.metadata import validate_metadata

from goldilocks_ml.hashing import is_sha256, sha256_file

PSDI_API = "https://data-collections.psdi.ac.uk/api"
# Kept as a literal rather than imported from goldilocks_ml.inference so that
# publishing stays independent of the serving side.
MODEL_RECORD_FILE = "model.json"
RESERVED_FILE_NAMES = frozenset(
    {"README.md", "manifest.json", "metadata.json", MODEL_RECORD_FILE}
)


@dataclass(frozen=True, slots=True)
class Artifact:
    """A model artifact and its expected integrity metadata."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Deposition:
    """Validated metadata and files ready for one PSDI draft."""

    directory: Path
    metadata: dict[str, Any]
    community: str
    artifacts: tuple[Artifact, ...]
    files: dict[str, Path]


class DraftCleanupError(RuntimeError):
    """An upload failed and the resulting partial draft could not be deleted."""

    def __init__(
        self,
        draft_id: str | None,
        upload_error: Exception,
        cleanup_error: Exception,
    ) -> None:
        self.draft_id = draft_id
        self.upload_error = upload_error
        self.cleanup_error = cleanup_error
        draft_label = draft_id or "unknown"
        super().__init__(
            f"PSDI draft {draft_label} upload failed ({upload_error}); "
            f"cleanup also failed ({cleanup_error}); remove the partial draft in PSDI"
        )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _artifact_from_dict(value: object) -> Artifact:
    if not isinstance(value, dict):
        raise ValueError("each manifest artifact must be an object")
    try:
        name = value["name"]
        size_bytes = value["size_bytes"]
        sha256 = value["sha256"]
    except KeyError as error:
        raise ValueError(f"artifact is missing {error.args[0]!r}") from error
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("artifact names must be non-empty basenames")
    if not isinstance(size_bytes, int) or size_bytes < 0:
        raise ValueError(f"invalid size for {name}")
    if not is_sha256(sha256):
        raise ValueError(f"invalid SHA-256 for {name}")
    return Artifact(name=name, size_bytes=size_bytes, sha256=sha256)


def describe_artifact(path: Path) -> Artifact:
    """Return the manifest entry for one local artifact."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return Artifact(
        name=path.name,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the PSDI base schema plus its omitted contributors field."""
    candidate = deepcopy(metadata)
    record_files = candidate.get("files")
    default_preview = None
    if isinstance(record_files, dict):
        default_preview = record_files.pop("default_preview", None)
        if default_preview is not None and (
            not isinstance(default_preview, str)
            or not default_preview
            or Path(default_preview).name != default_preview
        ):
            raise ValueError("files.default_preview must be a non-empty basename")
    record_metadata = candidate.get("metadata")
    if not isinstance(record_metadata, dict):
        raise ValueError("metadata must contain a metadata object")
    contributors = record_metadata.pop("contributors", None)
    validated = validate_metadata(candidate, "base")
    if default_preview is not None:
        validated["files"]["default_preview"] = default_preview
    if contributors is None:
        return validated
    if not isinstance(contributors, list) or not contributors:
        raise ValueError("contributors must be a non-empty list")
    for contributor in contributors:
        if not isinstance(contributor, dict):
            raise ValueError("each contributor must be an object")
        person = contributor.get("person_or_org")
        if not isinstance(person, dict) or person.get("type") != "personal":
            raise ValueError("each contributor must be a personal person_or_org")
        for field in ("name", "given_name", "family_name"):
            if not isinstance(person.get(field), str) or not person[field]:
                raise ValueError(f"each contributor must have a non-empty {field}")
        role = contributor.get("role")
        if not isinstance(role, dict) or not isinstance(role.get("id"), str):
            raise ValueError("each contributor must have a role id")
        affiliations = contributor.get("affiliations")
        if not isinstance(affiliations, list) or not affiliations:
            raise ValueError("each contributor must have affiliations")
        if any(
            not isinstance(affiliation, dict)
            or not isinstance(affiliation.get("name"), str)
            or not affiliation["name"]
            for affiliation in affiliations
        ):
            raise ValueError("each contributor affiliation must have a name")
    validated["metadata"]["contributors"] = contributors
    return validated


def find_markdown_tables(text: str) -> list[int]:
    """Return the line numbers of any GitHub-flavoured table delimiter rows.

    PSDI renders the model card with a plain Markdown previewer, which has no
    table extension: a table arrives as a wall of pipes and dashes. Every
    deposit so far has laid its numbers out as aligned columns inside a fenced
    block instead, which renders anywhere, and this keeps the next one from
    finding out the hard way after upload.

    Fenced blocks are skipped, so a card is free to draw a table inside one.
    """
    lines: list[int] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
            continue
        if fenced or "|" not in stripped or "-" not in stripped:
            continue
        if all(character in "|-: \t" for character in stripped):
            lines.append(number)
    return lines


def load_deposition(directory: Path, artifact_directory: Path) -> Deposition:
    """Load metadata and verify every upload file before any network mutation."""
    directory = directory.resolve()
    artifact_directory = artifact_directory.resolve()
    manifest_path = directory / "manifest.json"
    metadata_path = directory / "metadata.json"
    readme_path = directory / "README.md"
    record_path = directory / MODEL_RECORD_FILE

    manifest = _load_json(manifest_path)
    metadata = _load_json(metadata_path)
    metadata = _validate_metadata(metadata)

    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    community = manifest.get("community")
    if not isinstance(community, str) or not community:
        raise ValueError("manifest community must be a non-empty string")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("manifest artifacts must be a non-empty list")
    artifacts = tuple(_artifact_from_dict(value) for value in raw_artifacts)
    artifact_names = [artifact.name for artifact in artifacts]
    if len(artifact_names) != len(set(artifact_names)):
        raise ValueError("manifest artifact names must be unique")
    collisions = sorted(set(artifact_names) & RESERVED_FILE_NAMES)
    if collisions:
        names = ", ".join(collisions)
        raise ValueError(f"manifest artifact names are reserved upload files: {names}")
    inference_requirements = manifest.get("inference_requirements")
    if not isinstance(inference_requirements, dict) or not inference_requirements:
        raise ValueError("manifest inference_requirements must be a non-empty object")

    if not readme_path.is_file():
        raise FileNotFoundError(readme_path)
    table_lines = find_markdown_tables(readme_path.read_text(encoding="utf-8"))
    if table_lines:
        numbers = ", ".join(str(number) for number in table_lines)
        raise ValueError(
            f"{readme_path.name} uses Markdown tables (line(s) {numbers}), which "
            "the record page cannot render; lay the values out as aligned "
            "columns inside a fenced block instead"
        )
    # Without this a deposit is a file nobody can load: the digests, the
    # feature contract, and the target contract exist only in prose. Both
    # records published before it existed needed one reconstructed afterwards.
    if not record_path.is_file():
        raise FileNotFoundError(
            f"{record_path} is missing; a deposit needs {MODEL_RECORD_FILE} so "
            "the artifact can be loaded rather than only described"
        )
    files = {
        "README.md": readme_path,
        "manifest.json": manifest_path,
        MODEL_RECORD_FILE: record_path,
    }
    for artifact in artifacts:
        path = artifact_directory / artifact.name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if actual_size != artifact.size_bytes:
            raise ValueError(
                f"{artifact.name} has {actual_size} bytes; "
                f"expected {artifact.size_bytes}"
            )
        actual_digest = sha256_file(path)
        if actual_digest != artifact.sha256:
            raise ValueError(
                f"{artifact.name} SHA-256 is {actual_digest}; "
                f"expected {artifact.sha256}"
            )
        files[artifact.name] = path

    default_preview = metadata.get("files", {}).get("default_preview")
    if default_preview is not None and default_preview not in files:
        raise ValueError(f"default preview is not an upload file: {default_preview}")

    return Deposition(
        directory=directory,
        metadata=metadata,
        community=community,
        artifacts=artifacts,
        files=files,
    )


def read_token(path: Path) -> str:
    """Read a non-empty token from a file inaccessible to group and other users."""
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"token file permissions must be 600 or stricter, not {mode:o}"
        )
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("token file is empty")
    return token


def require_upload_confirmation(*, confirm_upload: bool) -> None:
    """Require explicit confirmation before creating a remote PSDI draft."""
    if not confirm_upload:
        raise ValueError("upload requires --confirm-upload")


def create_deposition(
    deposition: Deposition,
    *,
    token: str,
    repository_factory: Callable[..., Any] = InvenioRepository,
) -> str:
    """Create, populate, and bind one PSDI draft without submitting it."""
    repository = repository_factory(url=PSDI_API, api_key=token)
    draft = repository.depositions.create()
    draft_id = None
    try:
        draft_id = draft.get()["id"]
        if not isinstance(draft_id, str) or not draft_id:
            raise ValueError("PSDI draft response has no valid id")
        draft.update(deposition.metadata)
        draft.files.upload(deposition.files)
        draft.bind(deposition.community)
    except Exception as upload_error:
        try:
            draft.delete()
        except Exception as cleanup_error:
            raise DraftCleanupError(
                draft_id,
                upload_error,
                cleanup_error,
            ) from upload_error
        raise
    return draft_id


def _add_remote_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--token-file",
        type=Path,
        required=True,
        help="path to a token file with mode 600 or stricter",
    )
    parser.add_argument(
        "--confirm-upload",
        action="store_true",
        help="confirm creation of a real draft on PSDI",
    )


def add_parser(groups: argparse._SubParsersAction) -> None:
    """Register the ``publish`` group on the shared command line."""
    parser = groups.add_parser(
        "publish",
        help="validate and upload a model deposit to PSDI Data Collections",
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(handler=_run)
    subparsers = parser.add_subparsers(dest="publish_command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate metadata and artifact integrity offline"
    )
    validate.add_argument("deposition", type=Path)
    validate.add_argument("--artifact-directory", type=Path, required=True)

    checksum = subparsers.add_parser(
        "checksum", help="print a manifest entry for one local artifact"
    )
    checksum.add_argument("artifact", type=Path)

    upload = subparsers.add_parser(
        "upload", help="validate and create a new PSDI draft"
    )
    upload.add_argument("deposition", type=Path)
    upload.add_argument("--artifact-directory", type=Path, required=True)
    _add_remote_arguments(upload)

    return parser


def _run(args: argparse.Namespace) -> None:
    """Carry out one publish command."""
    if args.publish_command == "checksum":
        artifact = describe_artifact(args.artifact)
        print(
            json.dumps(
                {
                    "name": artifact.name,
                    "size_bytes": artifact.size_bytes,
                    "sha256": artifact.sha256,
                },
                indent=2,
            )
        )
        return

    deposition = load_deposition(args.deposition, args.artifact_directory)
    if args.publish_command == "validate":
        names = ", ".join(deposition.files)
        print(f"Valid deposition for {deposition.community}: {names}")
        return

    require_upload_confirmation(confirm_upload=args.confirm_upload)
    token = read_token(args.token_file)
    draft_id = create_deposition(
        deposition,
        token=token,
    )
    print(f"Created and bound PSDI draft {draft_id}; review not submitted")
