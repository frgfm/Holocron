# Copyright (C) 2024-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///

import logging
import re
import sys
import tomllib
from pathlib import Path

import yaml

DOCKERFILES = ["api/Dockerfile"]
PYPROJECTS = ["./pyproject.toml", "api/pyproject.toml"]
REQUIREMENTS_TXT = ["demo/requirements.txt"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(levelname)s:     %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


def parse_dep_str(dep_str: str) -> dict[str, str]:
    # No version specified
    if all(dep_str.find(char) == -1 for char in ("<", ">", "=")):
        version_idx = len(dep_str)
    else:
        version_idx = min(idx for idx in (dep_str.find(char) for char in ("<", ">", "=")) if idx != -1)
    pkg_idx = dep_str.find("[")
    extra_idx = dep_str.find("]")
    return {
        "pkg": dep_str[: pkg_idx if pkg_idx != -1 else version_idx],
        "version": dep_str[version_idx:],
        "extras": [] if extra_idx == -1 else dep_str[pkg_idx + 1 : extra_idx].split(","),
    }


def main():  # noqa: PLR0912
    # Retrieve & parse all deps files
    deps_dict = {
        "uv": [],
        "ruff": [],
        "ty": [],
        "pre-commit": [],
        "pytest": [],
        "pytest-cov": [],
        "pytest-pretty": [],
        "onnxruntime": [],
        "huggingface-hub": [],
        "numpy": [],
        "Pillow": [],
    }
    # Parse dockerfiles
    for dockerfile in DOCKERFILES:
        dockerfile_content_ = Path(dockerfile).read_text(encoding="utf-8")
        uv_version = re.search(r"ghcr\.io/astral-sh/uv:(\d+\.\d+\.\d+)", dockerfile_content_).group(1)  # ty: ignore[possibly-unbound-attribute]
        deps_dict["uv"].append({"file": dockerfile, "version": f"=={uv_version}"})
    # Parse pyproject.toml
    for pyproject_path in PYPROJECTS:
        with Path(pyproject_path).open("rb") as f:
            pyproject = tomllib.load(f)

        optional_deps = pyproject["project"]["optional-dependencies"]
        for requirements in (
            pyproject["project"]["dependencies"],
            optional_deps.get("quality", []),
            optional_deps.get("test", []),
        ):
            parsed_deps = [parse_dep_str(dep) for dep in requirements]
            parsed_deps = {dep["pkg"]: dep for dep in parsed_deps}
            for dep in deps_dict:  # noqa: PLC0206
                if dep in parsed_deps:
                    deps_dict[dep].append({"file": pyproject_path, "version": parsed_deps[dep]["version"]})

    # Parse requirements.txt
    for req_txt in REQUIREMENTS_TXT:
        with Path(req_txt).open("r", encoding="utf-8") as f:
            reqs = f.readlines()
        txt_deps = [parse_dep_str(dep.strip("\n")) for dep in reqs]
        txt_deps = {dep["pkg"]: dep for dep in txt_deps}
        for dep in deps_dict:  # noqa: PLC0206
            if dep in txt_deps:
                deps_dict[dep].append({"file": req_txt, "version": txt_deps[dep]["version"]})

    # Parse github/workflows/...
    for workflow_file in Path(".github/workflows").glob("*.yml"):
        with workflow_file.open("r") as f:
            workflow = yaml.safe_load(f)
            if "env" in workflow and "UV_VERSION" in workflow["env"]:
                deps_dict["uv"].append({
                    "file": str(workflow_file),
                    "version": f"=={workflow['env']['UV_VERSION'].lstrip('v')}",
                })

    # Assert all deps are in sync
    troubles = []
    for dep, versions in deps_dict.items():
        versions_ = {v["version"] for v in versions}
        if len(versions_) > 1:
            inv_dict = {v: set() for v in versions_}
            for version in versions:
                inv_dict[version["version"]].add(version["file"])
            troubles.extend([
                f"\033[31m{dep}\033[0m:",
                "\n".join(f"- '{v}': {', '.join(files)}" for v, files in inv_dict.items()),
            ])

    if len(troubles) > 0:
        raise AssertionError("Some dependencies are out of sync:\n\n" + "\n".join(troubles))
    logger.info("\033[32mAll dependencies are in sync!\033[0m")


if __name__ == "__main__":
    main()
