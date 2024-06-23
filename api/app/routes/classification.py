# Copyright (C) 2022-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


from fastapi import APIRouter, File, UploadFile, status

from app.schemas import ClsCandidate
from app.vision import CLF_CFG, classify_image, decode_image

router = APIRouter()


@router.post("/", status_code=status.HTTP_200_OK, summary="Perform image classification")
def classify(file: UploadFile = File(...)) -> ClsCandidate:
    """Runs holocron vision model to analyze the input image"""
    probs = classify_image(decode_image(file.file.read()))

    return ClsCandidate(
        value=CLF_CFG["classes"][probs.argmax()],
        confidence=float(probs.max()),
    )
