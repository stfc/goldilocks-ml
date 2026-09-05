# Deposit format

Publishing a model means publishing two kinds of thing: the model file, which is
large and stays out of Git, and four small files describing it, which are
committed.

```text
deposits/<setting>/<quantity>/<family>/
├── README.md       # model card, and the record's preview page
├── manifest.json   # which bytes, and what is needed to load them
├── model.json      # what the artifact is, machine-readable
└── metadata.json   # PSDI discovery metadata — sent, not uploaded
```

The first three are uploaded alongside the artifacts. `metadata.json` is what
PSDI is *told*, not a file it receives.

## `model.json`

Copy it from your run's `model/` folder — the training run writes it. It is what
makes a record loadable rather than only described, and publishing refuses a
deposit without one.

For an artifact fitted before that was true, write one afterwards and set
`"record_origin": "reconstructed"`.

`role` says what the artifact is for:

| Role | Meaning |
| --- | --- |
| `model` | answers a question; `load_model` serves it |
| `feature_extractor` | supplies input to another model; `load_model` declines it by name |

## `manifest.json`

```json
{
  "schema_version": 1,
  "community": "data-to-knowledge",
  "artifacts": [
    {"name": "model.bin", "size_bytes": 12345, "sha256": "64 lowercase hex chars"}
  ],
  "inference_requirements": {
    "artifact_format": "describe the serializer",
    "feature_contract": "versioned-feature-schema",
    "target": "prediction_target"
  }
}
```

Generate every artifact entry with `goldilocks-ml publish checksum PATH` rather
than typing sizes and digests. Names must be basenames, and cannot be
`README.md`, `manifest.json` or `metadata.json`.

`inference_requirements` is free-form and is for whoever loads the file later:
the serializer, the library versions that matter, the feature schema, the
prediction target, and any support files.

## `metadata.json`

Copy the closest existing deposit and check these:

| Field | Purpose |
| --- | --- |
| `access` | public record and files |
| `files.default_preview` | `README.md`, so the record page shows the card |
| `metadata.title` | specific model and task |
| `metadata.description` | short HTML shown in search results |
| `metadata.creators` | who the citation names |
| `metadata.contributors` | other project roles |
| `metadata.rights` | the licence on the artifact |
| `metadata.resource_type` | `model` |
| `metadata.version` | the release version |
| `custom_fields.dsmd` | `[]` — see below |

PSDI only stores `custom_fields.dsmd` keys it has registered, and drops the rest
silently with a 200. None of ours are registered, so send an empty list rather
than a block that looks stored and is not.

Creators and contributors do different jobs: PSDI builds the citation from
`creators`, so do not add the whole project there.

## `README.md`

The model card should let a colleague decide whether and how to use the model
without opening anything else:

1. what it predicts, and in what units or convention;
2. how to load it, and the library versions that matter;
3. how good it is, measured on held-out data;
4. where it fails, and what it has not been tested on;
5. what the training data was;
6. checksum verification, and an unsafe-deserialization warning for a pickle.

Do not claim metrics, dataset digests or guarantees that cannot be tied to the
exact released artifact.

**No Markdown tables.** PSDI previews the card with a plain Markdown renderer
that has no table extension, so a table arrives as a wall of pipes. Publishing
refuses a card containing one. Lay numbers out as aligned columns inside a
fenced block, which renders anywhere. A table *inside* a fenced block is fine.
