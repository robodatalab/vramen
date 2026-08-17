#!/usr/bin/env bash
# Cut a release: bump the version, commit, tag, push, create the GitHub
# release. The publish workflow reacts to the release and uploads to PyPI.
#
# Any failure rolls the repository back to exactly where it started -- local
# and remote -- so a half-finished release never survives. Re-running after a
# rollback therefore produces the same version again, not the next one.
#
# Usage: publish.sh [major|minor|patch]   (default: minor)

set -euo pipefail

PART="${1:-minor}"
PYTHON="${PYTHON:-python3}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() {
	echo "publish: $*" >&2
	exit 1
}

# --- preflight -------------------------------------------------------------
# Nothing here has changed any state yet, so these exit without a rollback.

command -v gh >/dev/null || die "gh is not installed"
[ -z "$(git status --porcelain)" ] || die "working tree is dirty -- commit or stash first"

branch="$(git rev-parse --abbrev-ref HEAD)"
[ "$branch" = "main" ] || die "publish from main, not $branch"

git fetch --quiet origin main || die "could not reach origin"
BASE="$(git rev-parse HEAD)"
[ "$BASE" = "$(git rev-parse origin/main)" ] || die "main and origin/main disagree -- pull or push first"

# --- rollback --------------------------------------------------------------
# Each step sets its flag only once it has succeeded, so the rollback undoes
# precisely what happened and nothing more. Every undo is best-effort: one
# that fails reports itself and the rest still run.

TAG=""
committed=0
tagged=0
pushed_main=0
pushed_tag=0

rollback() {
	local status=$?
	[ "$status" -eq 0 ] && return 0

	echo >&2
	echo "publish: failed -- rolling back to ${BASE:0:12}" >&2

	# A release may exist even when gh reported an error, and deleting a tag
	# out from under one would strand it.
	if [ -n "$TAG" ] && gh release view "$TAG" >/dev/null 2>&1; then
		gh release delete "$TAG" --yes >/dev/null 2>&1 \
			&& echo "  deleted release $TAG" >&2 \
			|| echo "  COULD NOT delete release $TAG -- remove it by hand" >&2
	fi

	if [ "$pushed_tag" = 1 ]; then
		git push --quiet --delete origin "$TAG" 2>/dev/null \
			&& echo "  deleted remote tag $TAG" >&2 \
			|| echo "  COULD NOT delete remote tag $TAG -- remove it by hand" >&2
	fi

	if [ "$pushed_main" = 1 ]; then
		# --force-with-lease refuses if anyone else pushed in the meantime,
		# in which case their commit is the one worth keeping.
		git push --quiet --force-with-lease origin "$BASE:main" 2>/dev/null \
			&& echo "  rewound origin/main to ${BASE:0:12}" >&2 \
			|| echo "  COULD NOT rewind origin/main -- it still has the release commit" >&2
	fi

	if [ "$tagged" = 1 ]; then
		git tag -d "$TAG" >/dev/null 2>&1 && echo "  deleted local tag $TAG" >&2
	fi

	if [ "$committed" = 1 ]; then
		git reset --quiet --hard "$BASE" && echo "  reset to ${BASE:0:12}" >&2
	else
		# The version bump may still be sitting in the working tree.
		git checkout --quiet -- pyproject.toml 2>/dev/null || true
	fi

	echo "publish: rolled back; nothing was released" >&2
}
trap rollback EXIT

# --- release ---------------------------------------------------------------

NEW="$("$PYTHON" scripts/bump_version.py "$PART")"
TAG="v$NEW"

if git rev-parse -q --verify "refs/tags/$TAG" >/dev/null 2>&1; then
	die "tag $TAG already exists locally"
fi
if git ls-remote --exit-code --tags origin "$TAG" >/dev/null 2>&1; then
	die "tag $TAG already exists on origin"
fi

echo "publish: releasing $TAG"

git add pyproject.toml
git commit --quiet -m "Release $TAG"
committed=1

git tag "$TAG"
tagged=1

git push --quiet origin main
pushed_main=1

git push --quiet origin "$TAG"
pushed_tag=1

gh release create "$TAG" --title "$TAG" --generate-notes

echo "publish: released $TAG -- the publish workflow now uploads it to PyPI"
