# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "fastapi>=0.143.0,<1.0.0",
#     "pydantic-settings>=2.0.0,<3.0.0",
#     "uvicorn>=0.23.0,<1.0.0",
#     "python-multipart>=0.0.27,<1.0.0",
#     "Pillow>=13.0.0",
#     "certifi>=2026.10.19",
#     "onnxruntime>=1.31.0,<2.0.0",
#     "huggingface-hub>=1.0.0,<2.0.0",
#     "numpy>=1.19.5,<3.0.0",
# ]
# ///

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import uvicorn
from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse

DEFAULT_REPO = "frgfm/rexnet1_0x"
DEFAULT_REVISION = "9f2b2d7e23dcf64a13a4df8abe4b8ca2afa973cb"
API_DIR = Path(__file__).resolve().parents[1] / "api"


def build_app(repo: str, revision: str | None):
    if repo != DEFAULT_REPO and revision is None:
        raise ValueError("--revision is required when --repo is not the default model")
    os.environ["CLF_HUB_REPO"] = repo
    os.environ["CLF_HUB_REVISION"] = revision or DEFAULT_REVISION
    sys.path.insert(0, str(API_DIR))

    api = importlib.import_module("app.main")
    vision = importlib.import_module("app.vision")
    app = api.app

    async def predictions(file: Annotated[UploadFile, File(...)]):
        image_bytes = await file.read()

        def classify_logits(data: bytes):
            image = vision.decode_image(data)
            return vision.CLF_ORT.run(None, {vision.CLF_ORT.get_inputs()[0].name: vision.preprocess_image(image)})[0][0]

        logits = await api.run_image_work(classify_logits, image_bytes)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        indexes = np.argsort(probabilities)[-3:][::-1]
        return [{"name": vision.CLF_CFG["classes"][i], "confidence": float(probabilities[i])} for i in indexes]

    app.add_api_route("/predictions", predictions, methods=["POST"])

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

    return app


def main():
    parser = argparse.ArgumentParser(description="Holocron image classification demo")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face model repository")
    parser.add_argument("--revision", help="Immutable Hugging Face commit SHA for a custom model")
    parser.add_argument("--port", type=int, default=8001, help="Port on which the webserver will be run")
    parser.add_argument("--host", default="127.0.0.1", help="Interface on which the webserver will listen")
    args = parser.parse_args()
    try:
        app = build_app(args.repo, args.revision)
    except ValueError as exc:
        parser.error(str(exc))

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
