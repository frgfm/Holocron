# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import os

PROJECT_NAME: str = "Holocron API template"
PROJECT_DESCRIPTION: str = "Template API for Computer Vision"
VERSION: str = "0.2.1.dev0"
DEBUG: bool = os.environ.get("DEBUG", "") != "False"
HUB_REPO: str = "frgfm/rexnet1_5x"
