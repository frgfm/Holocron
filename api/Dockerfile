FROM python:3.11-slim

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/app"

# Install curl
RUN apt-get -y update \
    && apt-get -y install curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /bin/uv

# copy requirements file
COPY requirements.txt /tmp/requirements.txt

# install dependencies
RUN uv pip install --no-cache --system -r /tmp/requirements.txt

# copy project
COPY app /app/app
