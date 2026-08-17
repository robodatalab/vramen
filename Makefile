# Release automation. `make publish` bumps the version, tags it and cuts a
# GitHub release; the publish workflow reacts to the release and does the
# actual PyPI upload via trusted publishing.

PART ?= minor
PYTHON ?= python3

.PHONY: help version publish

help:
	@echo "make version                           print the current version"
	@echo "make publish [PART=major|minor|patch]  bump, tag and release (default: minor)"

version:
	@$(PYTHON) -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])"

# A release cannot be taken back: PyPI refuses to reuse a version number even
# after a delete. The checks below are cheap next to that.
publish:
	@test -z "$$(git status --porcelain)" \
	  || { echo "working tree is dirty -- commit or stash first"; exit 1; }
	@test "$$(git rev-parse --abbrev-ref HEAD)" = "main" \
	  || { echo "publish from main, not $$(git rev-parse --abbrev-ref HEAD)"; exit 1; }
	@git fetch --quiet origin main
	@test "$$(git rev-parse HEAD)" = "$$(git rev-parse origin/main)" \
	  || { echo "main and origin/main disagree -- pull or push first"; exit 1; }
	@new=$$($(PYTHON) scripts/bump_version.py $(PART)) || exit 1; \
	  git rev-parse -q --verify "refs/tags/v$$new" >/dev/null \
	    && { git checkout -- pyproject.toml; echo "tag v$$new already exists"; exit 1; }; \
	  echo "releasing v$$new"; \
	  git add pyproject.toml && \
	  git commit -q -m "Release v$$new" && \
	  git push -q origin main && \
	  git tag "v$$new" && \
	  git push -q origin "v$$new" && \
	  gh release create "v$$new" --title "v$$new" --generate-notes
