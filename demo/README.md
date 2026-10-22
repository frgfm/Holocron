# Holocron image classification demo

The demo uses the bounded FastAPI image handling in `api/app`. From the repository root, run:

```shell
uv run demo/app.py
```

Open <http://127.0.0.1:8001> and upload a JPEG or PNG image. The page shows the three highest softmax scores. Requests use the same 10 MiB body limit, two-request concurrency limit, and image checks as the API.

The default model is `frgfm/rexnet1_0x` at commit `9f2b2d7e23dcf64a13a4df8abe4b8ca2afa973cb`. To select another Hub model, provide both its repository and immutable commit:

```shell
uv run demo/app.py --repo owner/model --revision "$COMMIT_SHA"
```

Set `COMMIT_SHA` to that repository's 40-character commit ID.

The server listens on `127.0.0.1`. For a public demo behind a proxy or in a container, use `--host 0.0.0.0`. Private deployments need an authenticated gateway before requests reach the backend; prevent direct access around that gateway.

Deploy the full repository, including `api/app`, `api/pyproject.toml`, and `demo/index.html`. This replaces the Gradio launcher: the latest Gradio release requires Pillow below 13 and cannot resolve with the patched Pillow dependency. A hosted Hugging Face Space needs its launcher and SDK configuration updated before this replacement is deployed. Run `make run-demo` to use port 3000 instead.
