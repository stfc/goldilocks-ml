# Deposit format

Publishing a model means publishing two different kinds of thing: the model
file itself, which is large and stays out of version control, and a few small
files describing it, which are committed alongside the code.

This page is what those small files have to contain.

## Directory layout

```text
deposits/<task>/<model>/
├── README.md       # model card and default preview
├── manifest.json   # artifact identity and usage requirements
└── metadata.json   # PSDI discovery and attribution metadata
```

The corresponding artifact directory contains only the files named in the
manifest. It can contain one model file or a model plus required support files.

## `manifest.json`

The manifest binds filenames to exact bytes:

```json
{
  "schema_version": 1,
  "community": "data-to-knowledge",
  "artifacts": [
    {
      "name": "model.bin",
      "size_bytes": 12345,
      "sha256": "64-lowercase-hexadecimal-characters"
    }
  ],
  "inference_requirements": {
    "artifact_format": "describe the serializer",
    "feature_contract": "versioned-feature-schema",
    "target": "prediction_target"
  }
}
```

Each artifact entry has three fields:

| Field | Meaning |
| --- | --- |
| `name` | Exact file basename that will be uploaded |
| `size_bytes` | Exact integer file size in bytes, not the rounded MB shown by a web page |
| `sha256` | 64-character lowercase digest calculated from the complete file contents |

Run `goldilocks-ml publish checksum PATH` on the final file to generate all three
values together. A model with multiple required files needs one artifact entry
per file.

Artifact names must be basenames, not paths. Generate each entry from the final
file with `goldilocks-ml publish checksum`; do not reuse a digest from an unverified or
re-serialized copy. Artifact names must not be `README.md`, `manifest.json`, or
`metadata.json`, which are reserved for the deposit sidecars.

`inference_requirements` records what is needed to use the model for inference.
It is not a PSDI field and it does not run any code. It tells a consumer what
must remain compatible when loading the artifact, such as:

- the serialization format and important library versions;
- the expected feature schema or preprocessing;
- the prediction target and output meaning;
- any support files that must be used with the model.

Its content is model-specific and is preserved in the uploaded manifest.

## `metadata.json`

The reviewed examples are the best schema reference:

- `deposits/k_points/k_distance/qrf/metadata.json`
- `deposits/metallicity/cgcnn/metadata.json`

At minimum, check these fields for every release:

| Field | Purpose |
| --- | --- |
| `access` | Public or restricted record and files |
| `files.default_preview` | Use `README.md` for a readable record landing page |
| `metadata.title` | Specific model and task name |
| `metadata.description` | Short HTML discovery description |
| `metadata.creators` | People shown in the generated citation |
| `metadata.contributors` | Other project roles and affiliations |
| `metadata.rights` | Licence that applies to the released artifact |
| `metadata.resource_type` | Use `model` for model releases |
| `metadata.publisher` | Publication service, currently `PSDI` |
| `metadata.publication_date` | Date associated with this artifact release |
| `metadata.version` | Human-readable model release version |
| `custom_fields.dsmd` | Model family, target, feature contract, and runtime facts |

Creators and contributors serve different purposes. PSDI constructs the
citation from `creators`; do not add every project member there unless that is
the intended citation. Keep factual project roles in `contributors`.

## `README.md`

The model card should let a colleague decide whether and how to use the model
without opening the training notebook. Include:

1. task and prediction target;
2. required input and feature ordering;
3. output meaning and units;
4. conversion or post-processing outside the artifact;
5. training data and label definition;
6. artifact format and exact load-bearing library versions;
7. checksum verification and unsafe-deserialization warning where relevant;
8. intended scope, failure modes, and provenance gaps.

Do not claim metrics, dataset hashes, code commits, or calibration guarantees
that cannot be tied to the exact released artifact.

## What is uploaded

The CLI uploads:

- `README.md`;
- `manifest.json`;
- every artifact listed in `manifest.json`.

`metadata.json` is sent as record metadata and is not uploaded as a file. This
is intentional: the rendered record is authoritative for discovery metadata,
while the manifest and model card travel with the binary release.
