# Template for your Vision API using Holocron

## Installation

Install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git), [uv](https://docs.astral.sh/uv/), and optionally [Docker](https://docs.docker.com/get-docker/). The locked environment installs the remaining dependencies.

## Usage

### Starting your web server

You will need to clone the repository first:
```shell
git clone https://github.com/frgfm/Holocron.git
cd Holocron
```
Then start the development server from the repository root:

```shell
make lock-backend
make uvicorn-backend
```
Once completed, your [FastAPI](https://fastapi.tiangolo.com/) server should be running on port 8080.

To build and run the container instead, use `make start-backend`. Run the API tests with `make test-backend`.

The API template has no built-in authentication. Do not expose its listener directly to the Internet. For a private deployment, bind it to a private interface and require HTTPS plus authentication at a trusted reverse proxy or gateway before it forwards requests; prevent direct access to the backend so clients cannot bypass that boundary. The API buffers request bodies up to 10 MiB before multipart parsing, returns 408 when reading takes over 10 seconds, and permits at most two concurrent parsing/inference requests per process (each worker has its own limit). It returns 503 when those slots are occupied. Only JPEG and PNG images up to 25 megapixels and 8192 pixels per side are accepted. Model files are loaded from one immutable Hugging Face commit. Set `CLF_HUB_REPO` to an `owner/model` ID and `CLF_HUB_REVISION` to a 40-character commit SHA when selecting another model. Invalid settings fail during startup before downloads.

### Documentation and swagger

FastAPI comes with many advantages including speed and OpenAPI features. Once the server is running, open the generated documentation at: http://localhost:8080/docs


### Using the routes

You will find detailed instructions in the live documentation when your server is up, but here are some examples to use your available API routes:

#### Image classification

Using the following image:
<img src="https://m.media-amazon.com/images/I/517Nh08xqkL._AC_SX425_.jpg" width="50%" height="50%">

with this snippet:

```python
import requests

with open("/path/to/your/img.jpg", "rb") as f:
    print(requests.post("http://localhost:8080/classification", files={"file": f}).json())
```

should yield
```json
{"value": "French horn", "confidence": 0.9685316681861877}
```
