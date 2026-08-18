#!/usr/bin/env python3
"""Work out the version this push releases, and write it into pyproject.toml.

Every push to main releases, so the version has to be settled without anyone
being asked. It comes from one of two places:

  - The version already in pyproject.toml, when it stands above the last
    released tag. Raising it by hand is how a minor or a major gets cut.
  - The next patch after the highest released tag, which is the ordinary case.

Counting from the tag rather than from the file is what stops a revert of a
release commit from walking the version backwards into a number PyPI has
already handed out.

With --check-pypi the choice is walked past anything the index already holds.
That is what makes an upload done by hand -- scripts/publish_local.sh ships
without tagging -- something the next push recovers from rather than collides
with.

Prints the decision, and appends `version=` and `bumped=` to $GITHUB_OUTPUT
when that is set, which is where the workflow reads them from.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"

# Anchored to the start of a line so it cannot match requires-python or a
# version specifier inside a dependency string. [project] is the first table
# in the file, so the first match is the project version.
VERSION = re.compile(r'^(version\s*=\s*")(\d+)\.(\d+)\.(\d+)(")$', re.MULTILINE)
TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

RELEASE_JSON = "https://pypi.org/pypi/{name}/{version}/json"

Version = tuple[int, int, int]


def released() -> Version:
    """The highest vX.Y.Z tag, or 0.0.0 in a repository that has none.

    Tags that are not plain releases are ignored rather than guessed at, so a
    `v2.0.0rc1` lying around cannot decide what ships.
    """
    listed = subprocess.run(
        ["git", "tag", "--list", "v*"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    matched = (TAG.match(line.strip()) for line in listed.splitlines())
    return max(
        (as_version(match) for match in matched if match is not None),
        default=(0, 0, 0),
    )


def as_version(match: re.Match[str]) -> Version:
    major, minor, patch = (int(group) for group in match.group(1, 2, 3))
    return major, minor, patch


def on_pypi(name: str, version: str) -> bool:
    """Whether the index already holds this version.

    PyPI refuses to reuse a version number even after a delete, so a collision
    is worth finding out about here rather than from a rejected upload.
    """
    try:
        request = urllib.request.Request(
            RELEASE_JSON.format(name=name, version=version),
            headers={"User-Agent": "vramen-release"},
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return bool(response.status == 200)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise


def render(version: Version) -> str:
    return ".".join(str(part) for part in version)


def emit(**values: str) -> None:
    """Hand the decision to the workflow, and to whoever reads the log."""
    for key, value in values.items():
        print(f"resolve-release: {key}={value}")

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            for key, value in values.items():
                handle.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide the version to release.")
    parser.add_argument(
        "--check-pypi",
        action="store_true",
        help="step past any version the index already holds",
    )
    args = parser.parse_args()

    text = PYPROJECT.read_text()
    match = VERSION.search(text)
    if match is None:
        print(f'no `version = "X.Y.Z"` line found in {PYPROJECT}', file=sys.stderr)
        return 1

    declared = (int(match.group(2)), int(match.group(3)), int(match.group(4)))
    last = released()

    if declared > last:
        # Raised by hand, so it is a minor or a major somebody meant. It ships
        # as it stands rather than being bumped a second time.
        chosen = declared
        why = f"pyproject.toml already stands above {render(last)}"
    else:
        major, minor, patch = last
        chosen = (major, minor, patch + 1)
        why = f"the next patch after {render(last)}"

    if args.check_pypi:
        name = tomllib.loads(text)["project"]["name"]
        while on_pypi(name, render(chosen)):
            print(
                f"resolve-release: {name} {render(chosen)} is already on PyPI",
                file=sys.stderr,
            )
            major, minor, patch = chosen
            chosen = (major, minor, patch + 1)
            why = "the first patch the index does not already hold"

    version = render(chosen)
    bumped = chosen != declared

    if bumped:
        PYPROJECT.write_text(
            f"{text[: match.start()]}{match.group(1)}{version}"
            f"{match.group(5)}{text[match.end() :]}"
        )

    print(f"resolve-release: releasing {version} -- {why}")
    emit(version=version, bumped="true" if bumped else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
