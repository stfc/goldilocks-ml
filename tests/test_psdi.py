"""Tests for safe PSDI model deposition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from goldilocks_ml.console import main
from goldilocks_ml.psdi import (
    PSDI_API,
    Deposition,
    DraftCleanupError,
    create_deposition,
    describe_artifact,
    load_deposition,
    read_token,
    require_upload_confirmation,
)


def _write_deposition(tmp_path: Path, payload: bytes = b"model") -> tuple[Path, Path]:
    deposition = tmp_path / "deposit"
    artifacts = tmp_path / "artifacts"
    deposition.mkdir()
    artifacts.mkdir()
    artifact = artifacts / "model.bin"
    artifact.write_bytes(payload)
    (deposition / "README.md").write_text("model card", encoding="utf-8")
    (deposition / "model.json").write_text(
        json.dumps({"record_schema_version": 1, "role": "model"}), encoding="utf-8"
    )
    (deposition / "metadata.json").write_text(
        json.dumps(
            {
                "custom_fields": {"dsmd": [{}]},
                "files": {"enabled": True, "default_preview": "README.md"},
                "metadata": {
                    "title": "Model",
                    "description": "Description",
                    "creators": [
                        {
                            "person_or_org": {
                                "name": "Yin, Junwen",
                                "type": "personal",
                            }
                        }
                    ],
                    "rights": [{"id": "cc-by-4.0"}],
                    "resource_type": {"id": "model"},
                    "version": "v1.0",
                },
            }
        ),
        encoding="utf-8",
    )
    (deposition / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "community": "data-to-knowledge",
                "artifacts": [
                    {
                        "name": artifact.name,
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
                "inference_requirements": {"artifact_format": "test bytes"},
            }
        ),
        encoding="utf-8",
    )
    return deposition, artifacts


def test_load_deposition_validates_metadata_and_artifacts(tmp_path: Path) -> None:
    directory, artifacts = _write_deposition(tmp_path)

    deposition = load_deposition(directory, artifacts)

    assert deposition.community == "data-to-knowledge"
    assert set(deposition.files) == {
        "README.md",
        "manifest.json",
        "model.json",
        "model.bin",
    }
    assert deposition.metadata["files"]["default_preview"] == "README.md"


def test_load_deposition_rejects_wrong_digest(tmp_path: Path) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    (artifacts / "model.bin").write_bytes(b"other")

    with pytest.raises(ValueError, match="bytes|SHA-256"):
        load_deposition(directory, artifacts)


def test_load_deposition_rejects_duplicate_artifact_names(tmp_path: Path) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].append(manifest["artifacts"][0])
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="artifact names must be unique"):
        load_deposition(directory, artifacts)


@pytest.mark.parametrize(
    "name", ["README.md", "manifest.json", "metadata.json", "model.json"]
)
def test_load_deposition_rejects_reserved_artifact_names(
    tmp_path: Path, name: str
) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][0]["name"] = name
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="reserved upload files"):
        load_deposition(directory, artifacts)


def test_load_deposition_rejects_missing_default_preview(tmp_path: Path) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["files"]["default_preview"] = "missing.md"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="default preview is not an upload file"):
        load_deposition(directory, artifacts)


def _valid_contributor() -> dict[str, object]:
    return {
        "person_or_org": {
            "type": "personal",
            "name": "Doe, Jane",
            "given_name": "Jane",
            "family_name": "Doe",
        },
        "role": {"id": "other"},
        "affiliations": [{"name": "STFC"}],
    }


def test_load_deposition_preserves_valid_contributors(tmp_path: Path) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["contributors"] = [_valid_contributor()]
    metadata_path.write_text(json.dumps(metadata))

    deposition = load_deposition(directory, artifacts)

    assert deposition.metadata["metadata"]["contributors"] == [_valid_contributor()]


@pytest.mark.parametrize(
    ("contributors", "message"),
    [
        ([], "non-empty list"),
        (["not-an-object"], "must be an object"),
        ([{"person_or_org": {"type": "organizational"}}], "personal"),
        (
            [{**_valid_contributor(), "person_or_org": {"type": "personal"}}],
            "non-empty name",
        ),
        ([{**_valid_contributor(), "role": {}}], "role id"),
        ([{**_valid_contributor(), "affiliations": []}], "affiliations"),
        (
            [{**_valid_contributor(), "affiliations": [{}]}],
            "affiliation must have a name",
        ),
    ],
)
def test_load_deposition_rejects_invalid_contributors(
    tmp_path: Path,
    contributors: object,
    message: str,
) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["metadata"]["contributors"] = contributors
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match=message):
        load_deposition(directory, artifacts)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("inference_requirements", {}, "inference_requirements"),
    ],
)
def test_load_deposition_rejects_invalid_manifest_contract(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        load_deposition(directory, artifacts)


def test_describe_artifact_returns_manifest_entry(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model.bin"
    artifact_path.write_bytes(b"model")

    artifact = describe_artifact(artifact_path)

    assert artifact.name == "model.bin"
    assert artifact.size_bytes == 5
    assert artifact.sha256 == hashlib.sha256(b"model").hexdigest()


def test_read_token_rejects_group_access(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("secret", encoding="utf-8")
    token.chmod(0o640)

    with pytest.raises(PermissionError, match="600 or stricter"):
        read_token(token)


def test_read_token_accepts_private_file(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("secret\n", encoding="utf-8")
    token.chmod(0o600)

    assert read_token(token) == "secret"


def test_upload_requires_explicit_confirmation() -> None:
    with pytest.raises(ValueError, match="--confirm-upload"):
        require_upload_confirmation(confirm_upload=False)

    require_upload_confirmation(confirm_upload=True)


class _FakeFiles:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def upload(self, files: dict[str, Path]) -> None:
        assert "model.bin" in files
        self.calls.append("upload")


class _FakeDraft:
    def __init__(
        self,
        calls: list[str],
        *,
        fail_get: bool = False,
        fail_upload: bool = False,
        fail_delete: bool = False,
    ) -> None:
        self.calls = calls
        self.fail_get = fail_get
        self.fail_upload = fail_upload
        self.fail_delete = fail_delete
        self.files = _FakeFiles(calls)
        if fail_upload:
            self.files.upload = self._fail_upload

    def _fail_upload(self, files: dict[str, Path]) -> None:
        raise RuntimeError("upload failed")

    def get(self) -> dict[str, str]:
        self.calls.append("get")
        if self.fail_get:
            raise RuntimeError("get failed")
        return {"id": "draft-1"}

    def update(self, metadata: dict[str, object]) -> None:
        self.calls.append("update")

    def bind(self, community: str) -> None:
        assert community == "data-to-knowledge"
        self.calls.append("bind")

    def delete(self) -> None:
        self.calls.append("delete")
        if self.fail_delete:
            raise RuntimeError("delete failed")


class _FakeDepositions:
    def __init__(self, draft: _FakeDraft, calls: list[str]) -> None:
        self._draft = draft
        self.calls = calls

    def create(self) -> _FakeDraft:
        self.calls.append("create")
        return self._draft


def _repository_factory(draft: _FakeDraft, calls: list[str]):
    def factory(*, url: str, api_key: str):
        assert url == PSDI_API
        assert api_key == "secret"
        return type("Repository", (), {"depositions": _FakeDepositions(draft, calls)})()

    return factory


def test_create_deposition_leaves_bound_draft_unsubmitted(tmp_path: Path) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    deposition = load_deposition(directory, artifacts)
    calls: list[str] = []
    draft = _FakeDraft(calls)

    draft_id = create_deposition(
        deposition,
        token="secret",
        repository_factory=_repository_factory(draft, calls),
    )

    assert draft_id == "draft-1"
    assert calls == ["create", "get", "update", "upload", "bind"]


def test_create_deposition_deletes_partial_draft_on_failure(tmp_path: Path) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    deposition = load_deposition(directory, artifacts)
    calls: list[str] = []
    draft = _FakeDraft(calls, fail_upload=True)

    with pytest.raises(RuntimeError, match="upload failed"):
        create_deposition(
            deposition,
            token="secret",
            repository_factory=_repository_factory(draft, calls),
        )

    assert calls == ["create", "get", "update", "delete"]


def test_create_deposition_deletes_draft_when_initial_read_fails(
    tmp_path: Path,
) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    deposition = load_deposition(directory, artifacts)
    calls: list[str] = []
    draft = _FakeDraft(calls, fail_get=True)

    with pytest.raises(RuntimeError, match="get failed"):
        create_deposition(
            deposition,
            token="secret",
            repository_factory=_repository_factory(draft, calls),
        )

    assert calls == ["create", "get", "delete"]


def test_create_deposition_reports_upload_and_cleanup_failures(tmp_path: Path) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    deposition = load_deposition(directory, artifacts)
    calls: list[str] = []
    draft = _FakeDraft(calls, fail_upload=True, fail_delete=True)

    with pytest.raises(DraftCleanupError, match="draft-1") as error:
        create_deposition(
            deposition,
            token="secret",
            repository_factory=_repository_factory(draft, calls),
        )

    assert calls == ["create", "get", "update", "delete"]
    assert error.value.draft_id == "draft-1"
    assert str(error.value.upload_error) == "upload failed"
    assert str(error.value.cleanup_error) == "delete failed"


def test_checksum_cli_prints_manifest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"model")
    del monkeypatch  # the parser takes its arguments directly

    main(["publish", "checksum", str(artifact)])

    assert json.loads(capsys.readouterr().out) == {
        "name": "model.bin",
        "size_bytes": 5,
        "sha256": hashlib.sha256(b"model").hexdigest(),
    }


def test_validate_cli_reports_verified_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    del monkeypatch  # the parser takes its arguments directly

    main(
        ["publish", "validate", str(directory), "--artifact-directory", str(artifacts)]
    )

    output = capsys.readouterr().out
    assert "Valid deposition for data-to-knowledge" in output
    assert "README.md, manifest.json, model.json, model.bin" in output


def test_upload_cli_creates_draft_without_submitting_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory, artifacts = _write_deposition(tmp_path)
    token_file = tmp_path / "psdi.token"
    token_file.write_text("secret", encoding="utf-8")
    token_file.chmod(0o600)
    calls: list[tuple[Deposition, str]] = []

    def fake_create(deposition: Deposition, *, token: str) -> str:
        calls.append((deposition, token))
        return "draft-2"

    monkeypatch.setattr("goldilocks_ml.psdi.create_deposition", fake_create)
    main(
        [
            "publish",
            "upload",
            str(directory),
            "--artifact-directory",
            str(artifacts),
            "--token-file",
            str(token_file),
            "--confirm-upload",
        ]
    )

    assert len(calls) == 1
    assert calls[0][0].community == "data-to-knowledge"
    assert calls[0][1] == "secret"
    assert capsys.readouterr().out == (
        "Created and bound PSDI draft draft-2; review not submitted\n"
    )


def test_a_deposit_without_a_loadable_record_is_refused(tmp_path: Path) -> None:
    deposition, artifacts = _write_deposition(tmp_path)
    (deposition / "model.json").unlink()

    with pytest.raises(FileNotFoundError, match="so the artifact can be loaded"):
        load_deposition(deposition, artifacts)


DEPOSITS = Path(__file__).parents[1] / "deposits"


@pytest.mark.parametrize(
    "deposit",
    sorted(path.parent for path in DEPOSITS.rglob("model.json")),
    ids=lambda path: "/".join(path.parts[-3:]),
)
def test_every_shipped_deposit_record_agrees_with_its_manifest(deposit: Path) -> None:
    record = json.loads((deposit / "model.json").read_text())
    manifest = json.loads((deposit / "manifest.json").read_text())

    assert record["record_schema_version"] == 1
    # Absent means "model", the same default the loader applies.
    assert record.get("role", "model") in {"model", "feature_extractor"}

    # Every digest the record pins must be one the manifest publishes, or the
    # record describes a file the deposit does not contain.
    published = {item["name"]: item["sha256"] for item in manifest["artifacts"]}
    artifacts = record["artifacts"]
    for name, digest in artifacts.items():
        if not name.endswith("_sha256"):
            continue
        file_name = artifacts[name.removesuffix("_sha256")]
        assert published[file_name] == digest, file_name
