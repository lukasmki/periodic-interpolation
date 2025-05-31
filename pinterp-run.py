#!/usr/bin/env python3
"""
Command line utility for running `interp_periodic`

Usage:
    python pinterp_run.py path/to/file
    python pinterp_run.py path/to/file --nframes 20
    python pinterp_run.py path/to/file --nframes 20 --output path/to/output
"""

import argparse
import sys
from pathlib import Path

from ase import io
from pinterp import interp_periodic


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a file with optional frame count."
    )
    parser.add_argument("file", type=Path, help="Path to the input file.")
    parser.add_argument(
        "--nframes",
        type=int,
        default=20,
        help="Number of interpolating frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print score information",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check if file exists
    if not args.file.is_file():
        print(f"Error: File '{args.file}' not found.", file=sys.stderr)
        sys.exit(1)

    # set output file if not given
    if args.output is None:
        if args.file.suffix:
            args.output = args.file.with_name(
                f"{args.file.stem}-path{args.file.suffix}"
            )
        else:
            args.output = args.file.with_name(f"{args.file.stem}-path")

    if args.verbose:
        print(f"Input file: {args.file}")
        print(f"Output file: {args.output}")

    # interp
    atoms = io.read(args.file, index=slice(None))
    path = interp_periodic(
        atoms[0],
        atoms[-1],
        num_images=args.nframes,
        verbose=args.verbose,
    )
    io.write(args.output, path)


if __name__ == "__main__":
    main()
