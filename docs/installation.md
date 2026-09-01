# Installation

Goldilocks ML is currently installed directly from its GitHub repository.

## Prerequisites

Install:

- [Git](https://git-scm.com/downloads);
- [uv](https://docs.astral.sh/uv/getting-started/installation/);
- Python 3.12 or newer.

## Install from GitHub

Clone the repository and create its locked environment:

```bash
git clone https://github.com/stfc/goldilocks-ml.git
cd goldilocks-ml
uv sync
```

`uv sync` installs Goldilocks ML and its required dependencies into a local
`.venv` managed by `uv`.

## Check it worked

```bash
uv run goldilocks-ml --help
```

You should see the two things this tool does: `train` and `publish`.

Training the real scientific models needs a larger set of libraries — PyTorch
and pymatgen among them — which are not installed by default because they are
slow to install and most people do not need them straight away:

```bash
uv sync --extra models
```

Continue with [Train a model](training/index.md) or
[Publish a model](publishing.md).
