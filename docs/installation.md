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

## Check the installation

Both command-line tools should display their help:

```bash
uv run goldilocks-train --help
uv run goldilocks-psdi --help
```

Continue with [Train a model](training/index.md) or
[Publish a model](getting-started.md).
