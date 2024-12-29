# Copyright (C) 2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Borrowed & adapted from https://github.com/pytorch/vision/blob/main/.github/process_commit.py
This script finds the merger responsible for labeling a PR by a commit SHA. It is used by the workflow in
'.github/workflows/pull-labels.yml'. If there exists no PR associated with the commit or the PR is properly labeled,
this script is a no-op.
Note: we ping the merger only, not the reviewers, as the reviewers can sometimes be external to torchvision
with no labeling responsibility, so we don't want to bother them.
"""

import os
from pathlib import Path
from typing import Any, Set, Tuple

import requests
import yaml


def query_repo(cmd: str, *, accept) -> Any:
    auth = {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.environ.get("GITHUB_TOKEN") else {}
    response = requests.get(
        f"https://api.github.com/repos/{cmd}",
        headers={"Accept": accept, **auth},
        timeout=5,
    )
    return response.json()


def get_pr_merger_and_labels(repo: str, pr_number: int) -> Tuple[str, Set[str]]:
    # See https://docs.github.com/en/rest/reference/pulls#get-a-pull-request
    data = query_repo(f"{repo}/pulls/{pr_number}", accept="application/vnd.github.v3+json")
    merger = data.get("merged_by", {}).get("login")
    labels = {label["name"] for label in data["labels"]}
    return merger, labels


def main(args):
    # Load the labels
    with Path(__file__).parent.joinpath(args.file).open("r") as f:
        labels = yaml.safe_load(f)
        primary = set(labels["primary"])
        secondary = set(labels["secondary"])
    # Retrieve the PR info
    merger, labels = get_pr_merger_and_labels(args.repo, args.pr)
    # Check if the PR is properly labeled
    # For a PR to be properly labeled it should have one primary label and one secondary label
    is_properly_labeled = bool(primary.intersection(labels) and secondary.intersection(labels))
    # If the PR is not properly labeled, ping the merger
    if isinstance(merger, str) and not is_properly_labeled:
        print(f"@{merger}")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="PR label checker", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("repo", type=str, help="Repo full name")
    parser.add_argument("pr", type=int, help="PR number")
    parser.add_argument("--file", type=str, help="Path to the labels file", default="pull-labels.yml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)