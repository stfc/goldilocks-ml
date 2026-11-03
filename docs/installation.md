# Installation

```bash
pip install goldilocks-ml
```

That installs everything, including PyTorch, pymatgen, mace and the rest of
the scientific stack the real models need. It's enough to [use a published
model](inference.md), `is_magnetic` included, and to train with the built-in
reference trainers or the real ones.

## Check it worked

```bash
goldilocks-ml --version
goldilocks-ml --help
```

You should see the two things this tool does: `train` and `publish`.

## Working on Goldilocks ML itself

Clone the repository and let [uv](https://docs.astral.sh/uv/) build the locked
environment:

```bash
git clone https://github.com/stfc/goldilocks-ml.git
cd goldilocks-ml
uv sync --group dev
uv run pytest
```

Commands in these docs are written as `uv run goldilocks-ml …` because they run
from a clone. With the package installed, drop the `uv run`.

Continue with [Use a model](inference.md) or [Train a model](training/index.md).
