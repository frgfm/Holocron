from datetime import datetime
from pathlib import Path

shebang = ["#!usr/bin/python\n"]
blank_line = "\n"

copyright_notice = [f"# Copyright (C) 2019-{datetime.now().year}, François-Guillaume Fernandez.\n"]
license_notice = [
    "# This program is licensed under the Apache License version 2.\n",
    "# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.\n"
]

# Define all header options
HEADERS = [
    shebang + [blank_line] + copyright_notice + [blank_line] + license_notice,
    copyright_notice + [blank_line] + license_notice
]

IGNORED_FILES = ["version.py", "__init__.py"]
FOLDERS = ["holocron", "scripts", "references"]


def main():

    invalid_files = []

    # For every python file in the repository
    for folder in FOLDERS:
        for source_path in Path(__file__).parent.parent.joinpath(folder).rglob('**/*.py'):
            if source_path.name not in IGNORED_FILES:
                # Parse header
                header_length = max(len(option) for option in HEADERS)
                current_header = []
                with open(source_path) as f:
                    for idx, line in enumerate(f):
                        current_header.append(line)
                        if idx == header_length - 1:
                            break
                # Validate it
                if not any(
                    "".join(current_header[:min(len(option), len(current_header))]) == "".join(option)
                    for option in HEADERS
                ):
                    invalid_files.append(source_path)
    if len(invalid_files) > 0:
        invalid_str = "\n- " + "\n- ".join(map(str, invalid_files))
        raise AssertionError(f"Invalid header in the following files:{invalid_str}")


if __name__ == "__main__":
    main()
