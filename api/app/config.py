# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import tomllib
from pathlib import Path

from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

__all__ = ["settings"]

meta_ = tomllib.load((Path(__file__).parent.parent / "pyproject.toml").open("rb"))


class Settings(BaseSettings):
    # State
    PROJECT_NAME: str = "Holocron API template"
    PROJECT_DESCRIPTION: str = "Template API for Computer Vision"
    VERSION: str = meta_["project"]["version"]
    DEBUG: bool = False
    CLF_HUB_REPO: str = Field(
        "frgfm/rexnet1_5x",
        json_schema_extra=[{"min_length": 2, "example": "frgfm/rexnet1_5x"}],
    )
    CLF_HUB_REVISION: str = "af5a939881795422fa97771e8a06b33e19f2dcab"

    @field_validator("CLF_HUB_REPO")
    @classmethod
    def validate_hub_repo(cls, value: str) -> str:
        try:
            validate_repo_id(value)
        except HFValidationError as exc:
            raise ValueError("CLF_HUB_REPO must be a valid Hub repository ID") from exc
        return value

    @field_validator("CLF_HUB_REVISION")
    @classmethod
    def validate_hub_revision(cls, value: str) -> str:
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("CLF_HUB_REVISION must be a 40-character immutable commit SHA")
        return value


settings = Settings()
