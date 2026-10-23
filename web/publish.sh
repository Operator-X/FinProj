#!/usr/bin/env bash
# Republish Margem to GitHub Pages (https://operator-x.github.io/margem/).
# Rebuilds the site from the pipeline artifacts, copies only the website into
# ~/margem-site (the public repo) and pushes. The URL, and so the QR codes, never change.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="${MARGEM_SITE_DIR:-$HOME/margem-site}"
cd "$ROOT" && uv run python -m engine.webexport
cp "$ROOT/web/site/index.html" "$ROOT/web/site/favicon.png" "$ROOT/web/site/.nojekyll" "$SITE/"
cd "$SITE"
if git diff --quiet && git diff --cached --quiet; then echo "No changes to publish."; exit 0; fi
git add -A && git commit -q -m "Update Margem web app" && git push -q
echo "Published: https://operator-x.github.io/margem/ (live in about a minute)"
