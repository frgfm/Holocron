# Template for your Vision API using Holocron

## Installation

You will only need to install [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Docker](https://docs.docker.com/get-docker/). The container environment will be self-sufficient and install the remaining dependencies on its own.

## Usage

### Starting your web server

You will need to clone the repository first:
```shell
git clone https://github.com/frgfm/Holocron.git
```
then from the repo root folder, you can start your container:

```shell
PORT=8050 docker-compose up -d --build
```
Once completed, your [FastAPI](https://fastapi.tiangolo.com/) server should be running on port 8050 (feel free to change this in the previous command).

### Documentation and swagger

FastAPI comes with many advantages including speed and OpenAPI features. For instance, once your server is running, you can access the automatically built documentation and swagger in your browser at: http://localhost:8050/docs


### Using the routes

You will find detailed instructions in the live documentation when your server is up, but here are some examples to use your available API routes:

#### Image classification

Using the following image:
<img src="https://m.media-amazon.com/images/I/517Nh08xqkL._AC_SX425_.jpg" width="50%" height="50%">

with this snippet:

```python
import requests
with open('/path/to/your/img.jpg', 'rb') as f:
    data = f.read()
print(requests.post("http://localhost:8050/classification", files={'file': data}).json())
```

should yield
```
{'value': 'French horn', 'confidence': 0.9186984300613403}
```
