"""
Borrowed & adapted from https://github.com/pytorch/vision/blob/main/.github/process_commit.py
This script finds the merger responsible for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pr-labels.yml'. If there exists no PR associated with the commit or the PR is properly labeled,
this script is a no-op.
Note: we ping the merger only, not the reviewers, as the reviewers can sometimes be external to torchvision
with no labeling responsibility, so we don't want to bother them.
"""

# For a PR to be properly labeled it should have one primary label and one secondary label

# Should specify the type of change
PRIMARY_LABELS = {
    "type: new feature",
    "type: bug",
    "type: enhancement",
    "type: misc",
}

# Should specify what has been modified
SECONDARY_LABELS = {
    "topic: documentation",
    "module: models",
    "module: nn",
    "module: ops",
    "module: optim",
    "module: trainer",
    "module: utils",
    "ext: docs",
    "ext: references",
    "ext: scripts",
    "ext: tests",
    "topic: build",
    "topic: ci",
}


def main(args):
    print(args.labels)
    is_properly_labeled = bool(PRIMARY_LABELS.intersection(args.labels) and SECONDARY_LABELS.intersection(args.labels))
    if not is_properly_labeled:
        print("False")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='PR label checker',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('labels', type=str, help='Hash of the commit')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
