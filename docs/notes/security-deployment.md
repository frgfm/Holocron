# Security deployment acceptance

## GitHub Actions and Pages

The documentation workflow now uploads a Pages artifact and deploys it with the
`github-pages` environment. Before removing the old `SSH_DEPLOY_KEY` repository
secret, an owner should set **Settings → Pages → Build and deployment → Source**
to **GitHub Actions**, then run the `docs` workflow on `main` and confirm the
published site. After that succeeds, remove `SSH_DEPLOY_KEY`; the workflow no
longer reads it. Keep the `github-pages` environment restricted to `main`. A
required reviewer can be added if deployments need manual approval.

Proposed owner changes, checked on 22 October 2026 (none applied):

| Setting | Current | Proposed |
| --- | --- | --- |
| Default `GITHUB_TOKEN` permission | Write | Read repository contents |
| Workflow-token pull request creation and review approval | Enabled | Disabled |
| Secret scanning | Disabled | Enabled |
| Secret scanning push protection | Disabled | Enabled |

Keep the fork pull request approval rule enabled for
outside contributors, and require approval for first-time contributors at a
minimum. These controls complement the workflow-level permissions in this
change; they do not replace them.

Disabling workflow-token pull request creation also blocks the annual copyright
workflow's `gh pr create` step. An owner must open its generated branch as a pull
request, or supply a narrowly scoped GitHub App token for that step. Do not
enable workflow-token review approval to restore this convenience.

Enable secret scanning and push protection under **Settings → Code security and
analysis** when the repository plan supports them. Review and resolve any
existing findings before enabling blocking push protection for contributors.

## Ongoing acceptance

The `security-dependencies` workflow audits resolved library, API, and demo
Python environments on dependency changes and weekly. It also scans the built
API image, including OS packages used by its CA store. A green result only
covers the scanned revisions and image; it does not show that a rebuilt image
was deployed or that a hosting platform's external CA configuration was
checked. Track source fixes, rebuilt artifacts, deployed artifacts, and closed
alerts as separate states.

## Release guidance

Prepare the next `0.3.x` patch release from the current `0.3.1.dev0` development
line. Release notes should call out the Pillow 13 requirement, the explicit
PyTorch 2.9 minimum paired with torchvision 0.24, bounded JPEG/PNG uploads, and
the one-process HTML demo launcher. Build fresh wheel and source artifacts,
verify a clean wheel install, and rebuild the API image before owner-approved
publication or deployment. Recheck resolved dependency audits and repository
alerts after release; a source pull request alone does not close deployment
acceptance.
