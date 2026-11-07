#!/usr/bin/env bash
# One-time repo bootstrap: pushes secrets from .secrets.env and applies branch-protection.json.
# Requires: the authenticated gh user to have admin rights on the repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS_FILE="$SCRIPT_DIR/../.secrets.env"
PROTECTION_FILE="$SCRIPT_DIR/branch-protection.json"

if ! gh auth status >/dev/null 2>&1; then
  gh auth login
fi

REPO=$(gh repo view --json nameWithOwner --jq .nameWithOwner)

if [ ! -f "$SECRETS_FILE" ]; then
  echo "::error::$SECRETS_FILE not found. Copy .secrets.env.example to .secrets.env and fill in real values first." >&2
  exit 1
fi

gh secret set --env-file "$SECRETS_FILE" --repo "$REPO"
echo "Secrets from $SECRETS_FILE pushed to ${REPO}."

# Branch protection (classic or rulesets) is only enforced on public repos on GitHub Free; for a
# private repository it answers 403 with an "Upgrade to GitHub Pro" message. That is a plan
# limit, not a misconfiguration, so it does not abort the rest of this one-time setup.
set +e
PROTECTION_OUTPUT=$(gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "repos/${REPO}/branches/main/protection" \
  --input "$PROTECTION_FILE" 2>&1)
PROTECTION_STATUS=$?
set -e

if [ "$PROTECTION_STATUS" -eq 0 ]; then
  echo "Branch protection applied to ${REPO}@main from ${PROTECTION_FILE}."
elif echo "$PROTECTION_OUTPUT" | grep -q "Upgrade to GitHub Pro"; then
  echo "::warning::Branch protection was not applied - ${REPO} is private, and this GitHub plan only enforces branch protection on public repos. Secrets are set and everything else (CI, auto-PR) still works without it." >&2
else
  echo "::warning::Branch protection was not applied (unexpected error below). Secrets are still set." >&2
  echo "$PROTECTION_OUTPUT" >&2
fi
