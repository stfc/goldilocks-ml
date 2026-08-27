# Scripts

One-off tooling that is not part of the installed package: conversions and
checks we run against our own data, which anyone else has no reason to run.
Everything that ships lives under `src/goldilocks_ml/`.

| Script | What it does |
| --- | --- |
| `goldilocks_to_snapshot.py` | Convert PSDI record 75959-bwa52 into a training snapshot |
| `verify_qrf95_features.py` | Score reproduced features with the published QRF95 model |

Run them from the repository root with `uv run python scripts/<name>.py`.
