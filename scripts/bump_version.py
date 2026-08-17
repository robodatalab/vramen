#!/usr/bin/env python3
"""Bump the version in pyproject.toml and print the new value.

Usage: bump_version.py [major|minor|patch]   (default: minor)

Rewrites the version in place rather than round-tripping the TOML, so
comments and formatting elsewhere in the file are left untouched.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

# Anchored to the start of a line so it cannot match requires-python or a
# version specifier inside a dependency string. [project] is the first table
# in the file, so the first match is the project version.
VERSION = re.compile(r'^(version\s*=\s*")(\d+)\.(\d+)\.(\d+)(")$', re.MULTILINE)

PARTS = ("major", "minor", "patch")


def main() -> int:
    part = sys.argv[1] if len(sys.argv) > 1 else "minor"
    if part not in PARTS:
        print(f"unknown part {part!r}: expected one of {', '.join(PARTS)}", file=sys.stderr)
        return 2

    text = PYPROJECT.read_text()
    match = VERSION.search(text)
    if match is None:
        print(f'no `version = "X.Y.Z"` line found in {PYPROJECT}', file=sys.stderr)
        return 1

    major, minor, patch = (int(group) for group in match.group(2, 3, 4))
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1

    bumped = f"{major}.{minor}.{patch}"
    PYPROJECT.write_text(
        f"{text[: match.start()]}{match.group(1)}{bumped}{match.group(5)}{text[match.end() :]}"
    )
    print(bumped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
