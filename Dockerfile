FROM python:3.8-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1


COPY ./pyproject.toml /tmp/pyproject.toml
COPY ./README.md /tmp/README.md
COPY ./setup.py /tmp/setup.py
COPY ./holocron /tmp/holocron

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e /tmp/.
