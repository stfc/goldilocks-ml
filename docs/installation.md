# Installation

```bash
pip install goldilocks-ml
```

That installs everything, including PyTorch, pymatgen and the rest of the
scientific stack the real models need. It's enough to [use a published
model](inference.md) and to train with the built-in reference trainers or
the real ones.

### The is_magnetic classifier needs one more, manual step

`is_magnetic` reads a frozen mMACE backbone's embedding, which needs `mace`
on top of the install above. There is no extra for it: the `mace-torch` this
needs is a research collaborator's fork (`CheukHinHoJerry/mace`, not the
upstream `ACEsuit/mace` package of the same name on PyPI), which has no PyPI
release at all -- and a package published to PyPI cannot declare a direct
git dependency in its own metadata regardless. Install it by hand:

```bash
pip install ase==3.28.0 e3nn==0.4.4 sphericart==1.0.9 sphericart-torch==1.0.9
pip install "mace-torch @ git+https://github.com/CheukHinHoJerry/mace.git@ac8ff4764122ced0d57198fe2f9ba170c9fcd16d"
```

See `deposits/magnetism/is_magnetic/mace_mlp/VENDORING_TODO.md` in the
repository for the full story on that pinned commit.

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
