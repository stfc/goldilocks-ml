---
name: use-uv
description: Use uv for all Python package management in this project. Use when installing dependencies, running Python commands, adding packages, building, or managing environments. Never use pip, venv, or pipx.
---

# Use uv

Use `uv` for every Python package and environment operation.

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -c "..."
uv add <package>
uv add --group dev <package>
uv build
```

- Do not activate a virtual environment manually.
- Do not use `pip`, `pipx`, `venv`, or `virtualenv`.
- Declare dependencies in `pyproject.toml` and commit `uv.lock` once the project
  has one.
- Use optional dependency groups for heavyweight or platform-specific ML
  stacks when they are not required by every workflow.
- If a command shown here does not yet exist in the repository, establish the
  corresponding configuration before treating it as a required check.
