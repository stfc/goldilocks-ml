"""The `goldilocks-ml` command line.

One binary, grouped by responsibility, matching the shape Goldilocks Core
settles on: a group names what you are working with, a command names what to do
to it. Training and publication are what this package is responsible for.
Inference has no command here on purpose -- the side that issues that command
is Core, and this package gives it a library.

Grouping also disambiguates a word that means two things. `train validate`
checks a protocol against a snapshot; `publish validate` checks a deposit
against its artifacts. As separate binaries the difference was invisible.
"""

from __future__ import annotations

import argparse
import sys

from goldilocks_ml import __version__
from goldilocks_ml import cli as training
from goldilocks_ml import psdi as publication

# Errors that carry an actionable message. A traceback would only bury it.
REPORTED = (
    ValueError,
    FileNotFoundError,
    FileExistsError,
    NotADirectoryError,
    PermissionError,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the parser with every group mounted."""
    parser = argparse.ArgumentParser(
        prog="goldilocks-ml",
        description="Train, evaluate, and publish Goldilocks models.",
    )
    parser.add_argument(
        "--version", action="version", version=f"goldilocks-ml {__version__}"
    )
    groups = parser.add_subparsers(dest="group", required=True)
    training.add_parser(groups)
    publication.add_parser(groups)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run one command."""
    args = build_parser().parse_args(argv)
    try:
        args.handler(args)
    except REPORTED as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
