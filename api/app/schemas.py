# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from pydantic import BaseModel, Field


class ClsCandidate(BaseModel):
    value: str = Field(..., example="Wookie")
    confidence: float = Field(..., gte=0, lte=1)
