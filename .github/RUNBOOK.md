# Runbook: setting up this `.github` on GitHub

What lives here:

| Piece | What it does |
|---|---|
| `workflows/auto-pr.yml` | A push to `feature/**` opens a PR to `main` (if one isn't already open) |
| `workflows/ci.yml` | On a PR to `main`: `version-check`, `test`, `library-install`. On a push to `main`: the same, then `tag-release` |
| `actions/version-check` | Fails a PR that changed `src/`, `pyproject.toml` without bumping `version` in `pyproject.toml`, or that bumps it without changing `CHANGELOG.md` |
| `actions/tag-package` | Tags `main` as `dlkit-v<version>` once, if that tag doesn't exist |
| `setup/bootstrap.sh` | One-time: pushes `.secrets.env` as repo secrets, applies `branch-protection.json` |

`.secrets.env` is gitignored - never commit it.

## The first push

In **Git Bash** (in PowerShell, `bash` resolves to the WSL launcher, where `gh` is not logged in):

1. **The remote and `main`** (a root commit with no files):
   ```bash
   git init -b main
   gh repo create torchstack64 --private --source . --remote origin
   git commit --allow-empty -m "chore: initial commit"
   git push -u origin main
   ```

2. **The token.** A fine-grained PAT with access to this repository: **Pull requests: Read and
   write**, **Contents: Read-only**. It has to be a PAT: a PR opened with the default
   `GITHUB_TOKEN` doesn't trigger other workflows, so `ci.yml`'s `pull_request` event would never
   fire.
   ```bash
   cp .github/.secrets.env.example .github/.secrets.env
   # edit it: AUTO_PR_TOKEN=<the PAT>
   git check-ignore -v .github/.secrets.env     # MUST print the .gitignore rule; if not, stop
   ```

3. **Secrets and branch protection**
   ```bash
   bash .github/setup/bootstrap.sh
   gh secret list                                # AUTO_PR_TOKEN
   ```
   On a private repo under GitHub Free the protection step answers 403 ("Upgrade to GitHub Pro"):
   a plan limit, not a mistake. The script warns and carries on.

4. **The working branch.** CI opens a PR only from `feature/**`, so commits go to `feature/dev`:
   ```bash
   git switch -c feature/dev                                       # if you are not on it yet
   git add .
   git ls-files | grep -E "secrets\.env$|\.venv|egg-info|\.coverage"   # MUST be empty
   git commit -m "feat: <what>"
   git push -u origin feature/dev
   ```

5. **Watch the Actions tab.** `Auto PR to main` opens the PR; `CI` then runs in it.

6. **Merge** (merge commit), then `git switch main && git pull && git switch feature/dev && git merge main`.

## Later work

Keep committing on `feature/dev`: each push opens a PR, or updates the one that is open. When a
piece of work deserves its own PR, `git switch -c feature/<subject>` from `main` works the same
way. A PR that touches the package bumps `version` in `pyproject.toml`, the only
source of the version, and adds that version's section to `CHANGELOG.md`.

## If a job is added to `ci.yml`

Its id goes into `required_status_checks.contexts` in `setup/branch-protection.json`, applied again
with step 3; without that it runs but blocks nothing. If it must pass before a
release, it goes into the `needs` of `tag-release` too. The `Makefile` gets the same
command, so `make check` and the CI stay one claim, and the `workflows/ci.yml` row of the table at
the top gets the job's name.

## If a job in `ci.yml` is renamed

`setup/branch-protection.json` › `required_status_checks.contexts` has to match the job ids, and
be re-applied with step 3.
