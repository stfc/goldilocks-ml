"""Tests for safe PSDI model deposition."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from goldilocks_ml.psdi import (
    PSDI_API,
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
                "record_key": "test-model",
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
    assert set(deposition.files) == {"README.md", "manifest.json", "model.bin"}
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
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("record_key", "", "record_key"),
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
        fail_upload: bool = False,
    ) -> None:
        self.calls = calls
        self.fail_upload = fail_upload
        self.files = _FakeFiles(calls)
        if fail_upload:
            self.files.upload = self._fail_upload

    def _fail_upload(self, files: dict[str, Path]) -> None:
        raise RuntimeError("upload failed")

    def get(self) -> dict[str, str]:
        self.calls.append("get")
        return {"id": "draft-1"}

    def update(self, metadata: dict[str, object]) -> None:
        self.calls.append("update")

    def bind(self, community: str) -> None:
        assert community == "data-to-knowledge"
        self.calls.append("bind")

    def delete(self) -> None:
        self.calls.append("delete")


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
