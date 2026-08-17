#!/usr/bin/env bash
# Publish to PyPI straight from this machine. No git, no tags, no GitHub --
# nothing here talks to GitHub at all.
#
# This is the fallback for when the release-driven path (scripts/publish.sh,
# which needs GitHub to be up) is not an option. It trades away trusted
# publishing: uploads from a laptop need an API token and carry no PEP 740
# attestations, so prefer `make publish` when GitHub is healthy.
#
# Any failure restores the version and removes the artifacts it built.
#
# Usage: publish_local.sh [major|minor|patch|none]   (default: minor)
#          none -> publish the current version as-is, without bumping
#
# Credentials: UV_PUBLISH_TOKEN, read from .env (gitignored) if present, else
# from the environment; failing that, a [pypi] entry in ~/.pypirc.

set -euo pipefail

PART="${1:-minor}"
PYTHON="${PYTHON:-python3}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() {
	echo "publish-local: $*" >&2
	exit 1
}

pyproject_field() { # pyproject_field <key under [project]>
	"$PYTHON" -c "import tomllib,pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['$1'])"
}

# --- preflight -------------------------------------------------------------

command -v uv >/dev/null || die "uv is not installed"
[ -z "$(git status --porcelain)" ] \
	|| die "working tree is dirty -- commit or stash first, so what ships is what you reviewed"

# .env is gitignored and holds UV_PUBLISH_TOKEN. Sourced with -a so the token
# reaches uv as an environment variable; nothing here echoes it.
if [ -f .env ]; then
	set -a
	# shellcheck disable=SC1091
	. ./.env
	set +a
fi

if [ -z "${UV_PUBLISH_TOKEN:-}" ] && [ ! -f "$HOME/.pypirc" ]; then
	echo "publish-local: no UV_PUBLISH_TOKEN (checked .env and the environment) and" >&2
	echo "               no ~/.pypirc -- uv will fail unless credentials come from" >&2
	echo "               somewhere else, such as the keyring" >&2
fi

# --- rollback --------------------------------------------------------------

NAME="$(pyproject_field name)"
DIST_NAME="${NAME//-/_}" # wheels normalise dashes to underscores

bumped=0
built=0
VERSION=""

rollback() {
	local status=$?
	[ "$status" -eq 0 ] && return 0

	echo >&2
	echo "publish-local: failed -- rolling back" >&2

	if [ "$built" = 1 ] && [ -n "$VERSION" ]; then
		rm -f "dist/$DIST_NAME-$VERSION".tar.gz "dist/$DIST_NAME-$VERSION"-*.whl
		rmdir dist 2>/dev/null || true
		echo "  removed the artifacts for $VERSION" >&2
	fi

	if [ "$bumped" = 1 ]; then
		git checkout --quiet -- pyproject.toml \
			&& echo "  restored the version in pyproject.toml" >&2
	fi

	echo "publish-local: rolled back; nothing was uploaded" >&2
}
trap rollback EXIT

# --- version ---------------------------------------------------------------

if [ "$PART" = "none" ]; then
	VERSION="$(pyproject_field version)"
else
	VERSION="$("$PYTHON" scripts/bump_version.py "$PART")"
	bumped=1
fi

# PyPI will not reuse a version number even after a delete, and finding that
# out from a rejected upload is a worse way to learn it.
if curl -sf --max-time 15 "https://pypi.org/pypi/$NAME/$VERSION/json" >/dev/null 2>&1; then
	die "$NAME $VERSION is already on PyPI -- pick another version"
fi

echo "publish-local: publishing $NAME $VERSION"

# --- build and upload ------------------------------------------------------

uv build
built=1

uv publish

trap - EXIT
echo
echo "publish-local: uploaded $VERSION to PyPI"
if [ "$bumped" = 1 ]; then
	echo "publish-local: the version bump is uncommitted -- commit pyproject.toml"
fi
echo "publish-local: do NOT create a v$VERSION GitHub release; the workflow would"
echo "               try to upload $VERSION again and PyPI would reject it"
