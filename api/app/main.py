# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import asyncio
import time
from collections.abc import Callable
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from app.config import settings
from app.vision import CLF_CFG, classify_image, decode_image

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    debug=settings.DEBUG,
    version=settings.VERSION,
)

MAX_REQUEST_BYTES = 10 * 1024 * 1024
BODY_READ_TIMEOUT_SECONDS = 10
CLASSIFICATION_SLOTS = asyncio.Semaphore(2)


class BoundedRequestMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """Bound request bytes and active parsing/inference before route dispatch."""
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        content_lengths = [value for name, value in scope["headers"] if name == b"content-length"]
        rejection = None
        if len(content_lengths) > 1:
            rejection = JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        elif content_lengths:
            try:
                content_length = int(content_lengths[0])
                if content_length < 0:
                    rejection = JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
                elif content_length > MAX_REQUEST_BYTES:
                    rejection = JSONResponse({"detail": "Request body too large"}, status_code=413)
            except ValueError:
                rejection = JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        if rejection is not None:
            return await rejection(scope, receive, send)
        if CLASSIFICATION_SLOTS.locked():
            return await JSONResponse({"detail": "Server busy"}, status_code=503)(scope, receive, send)

        await CLASSIFICATION_SLOTS.acquire()
        try:
            body, status_code, detail = await self._read_body(receive)
            if status_code is not None:
                return await JSONResponse({"detail": detail}, status_code=status_code)(scope, receive, send)
            sent = False

            async def replay_receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay_receive, send)
        finally:
            CLASSIFICATION_SLOTS.release()
        return None

    @staticmethod
    async def _read_body(receive):
        body = bytearray()
        try:
            async with asyncio.timeout(BODY_READ_TIMEOUT_SECONDS):
                return await BoundedRequestMiddleware._collect_body(receive, body)
        except TimeoutError:
            return None, 408, "Request body timed out"

    @staticmethod
    async def _collect_body(receive, body):
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return None, 400, "Request body disconnected"
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                return None, 413, "Request body too large"
            body.extend(chunk)
            if not message.get("more_body", False):
                return bytes(body), None, None


app.add_middleware(BoundedRequestMiddleware)


class ProcessTimeMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """Append elapsed request time without changing ASGI cancellation behavior."""
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start_time = time.time()

        async def send_with_process_time(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-process-time", str(time.time() - start_time).encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_process_time)
        return None


app.add_middleware(ProcessTimeMiddleware)


class ClsCandidate(BaseModel):
    """Classification result"""

    value: str = Field(..., json_schema_extra=[{"example": "Wookie"}])
    confidence: float = Field(..., json_schema_extra=[{"gte": 0, "lte": 1}])


# Routing
@app.post("/classification", status_code=status.HTTP_200_OK, summary="Perform image classification")
async def classify(file: Annotated[UploadFile, File(...)]) -> ClsCandidate:
    """Run image classification and return the best class and confidence."""
    probs = await run_image_work(classify_uploaded_image_sync, file.file.read())
    return ClsCandidate(value=CLF_CFG["classes"][probs.argmax()], confidence=float(probs.max()))


async def run_image_work(function: Callable[[bytes], object], image_bytes: bytes):
    """Run image parsing and inference in a bounded thread.

    Raises:
        HTTPException: If the image is invalid or unsupported.
    """
    work = asyncio.create_task(asyncio.to_thread(function, image_bytes))
    finished = asyncio.Event()
    work.add_done_callback(lambda _work: finished.set())
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        while not work.done():
            try:
                await finished.wait()
            except asyncio.CancelledError:
                continue
        if not work.cancelled():
            work.exception()
        raise
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or unsupported image") from exc


def classify_uploaded_image_sync(image_bytes: bytes):
    return classify_image(decode_image(image_bytes))


class Status(BaseModel):
    status: str


# Healthcheck
@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def health_check() -> Status:
    return Status(status="ok")
