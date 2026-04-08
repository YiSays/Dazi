"""Entry point for `python -m dazi` and `dazi` CLI command."""

import argparse
import asyncio
import sys

from dazi import __version__
from dazi.main import run_repl


def main():
    parser = argparse.ArgumentParser(
        prog="dazi", description="DAZI — Develop Autonomously, Zero Interruption"
    )
    parser.add_argument("--version", action="version", version=f"v{__version__}")
    parser.parse_args()
    asyncio.run(run_repl())


if __name__ == "__main__":
    main()
