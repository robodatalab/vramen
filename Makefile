# Releasing is automatic: every push to main runs .github/workflows/publish.yml,
# which works out the version, commits it, tags it and uploads to PyPI via
# trusted publishing. Nothing here needs to be run to cut a release.
#
# What is left is the fallback for when GitHub is unavailable, and a way to ask
# what the current version is.

PART ?= patch
PYTHON ?= python3

.PHONY: help version next-version publish-local

help:
	@echo "make version                                 print the current version"
	@echo "make next-version                            print what the next push to main would release"
	@echo "make publish-local [PART=...|none]           fallback: bump, build and upload straight to PyPI"
	@echo "                                             PART=none publishes the current version as-is"
	@echo
	@echo "Releases happen on their own: push to main and the publish workflow"
	@echo "takes it from there."

version:
	@$(PYTHON) -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])"

# Answers without writing anything: the resolver only touches pyproject.toml
# when it decides to bump, and this restores it either way.
next-version:
	@cp pyproject.toml pyproject.toml.bak
	@$(PYTHON) scripts/resolve_release.py --check-pypi || true
	@mv pyproject.toml.bak pyproject.toml

# For when GitHub is down and the workflow cannot run. Uploads from this
# machine with an API token, so it forfeits trusted publishing and PEP 740
# attestations -- let the workflow do it whenever GitHub is healthy.
#
# This ships without tagging, so the version it uploads is one the repository
# has no record of. The next push to main notices: resolve_release.py asks the
# index and steps over anything already published.
publish-local:
	@PYTHON="$(PYTHON)" bash scripts/publish_local.sh $(PART)
