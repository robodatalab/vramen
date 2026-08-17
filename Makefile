# Release automation. `make publish` bumps the version, tags it and cuts a
# GitHub release; the publish workflow reacts to the release and does the
# actual PyPI upload via trusted publishing.
#
# A release cannot be taken back: PyPI refuses to reuse a version number even
# after a delete. So publish either completes or rolls the repository back to
# where it started -- see scripts/publish.sh.

PART ?= minor
PYTHON ?= python3

.PHONY: help version publish publish-local

help:
	@echo "make version                                 print the current version"
	@echo "make publish [PART=major|minor|patch]        bump, tag and release via GitHub (default: minor)"
	@echo "make publish-local [PART=...|none]           bump, build and upload straight to PyPI"
	@echo "                                             PART=none publishes the current version as-is"

version:
	@$(PYTHON) -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])"

publish:
	@PYTHON="$(PYTHON)" bash scripts/publish.sh $(PART)

# Fallback for when GitHub is unavailable. Uploads from this machine with an
# API token, so it forfeits trusted publishing and PEP 740 attestations --
# prefer `make publish` when GitHub is healthy.
publish-local:
	@PYTHON="$(PYTHON)" bash scripts/publish_local.sh $(PART)
