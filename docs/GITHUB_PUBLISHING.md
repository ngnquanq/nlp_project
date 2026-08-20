# GitHub publishing checklist

The GitHub repository must remain private and source-only. Local experiment
artifacts stay in this workspace but are excluded by `.gitignore`.

## Before staging

1. Run `make check-source` without relying on the restricted corpus.
2. Run `make check-private` locally when the corpus and saved artifacts are present.
3. Run a dedicated secret scanner over the working tree and any existing history.
4. Confirm `git status --ignored --short` marks restricted/generated paths with `!!`.
5. Confirm no candidate tracked file is larger than 50 MiB.

## Allowed source groups

Stage only reviewed paths such as `.github/`, `code/`, `configs/`, `docs/`,
`environments/`, `notebooks/`, `report/`, `slides/`, `tests/`, the root Markdown
files, Makefile, pytest configuration, requirements files, `.gitignore`, and
`.gitattributes`. Never use `git add .`, `git add -A`, or `git add --all`.

Do not stage the private specification DOCX, bundled paper PDF, corpora,
`artifacts/`, `checkpoint/` contents, predictions, generated metrics, `work/`,
`runs/`, or manual annotation records. Share required experiment artifacts through
the course-approved private channel with SHA-256 hashes.

Staging, committing, creating a remote, and pushing are separate operations and
must each be reviewed explicitly.
