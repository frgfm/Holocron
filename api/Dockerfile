FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-slim

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/app"

# copy requirements file
COPY api/requirements.txt /app/requirements.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /app/requirements.txt \
    && pip cache purge \
    && rm -rf /root/.cache/pip

# install holocron
COPY ./README.md /tmp/README.md
COPY ./setup.py /tmp/setup.py
COPY ./pyproject.toml /tmp/pyproject.toml
COPY ./holocron /tmp/holocron

RUN pip install -e /tmp/. \
    && pip cache purge \
    && rm -rf /root/.cache/pip

# copy project
COPY api/app /app/app
