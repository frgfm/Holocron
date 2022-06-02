FROM python:3.8.1-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1


COPY ./pyproject.toml /tmp/pyproject.toml
COPY ./README.md /tmp/README.md
COPY ./setup.py /tmp/setup.py
COPY ./holocron /tmp/holocron

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e /tmp/. \
    && pip cache purge \
    && rm -rf /root/.cache/pip
