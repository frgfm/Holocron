# Copyright (C) 2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import pytest_asyncio
import requests
from httpx import AsyncClient

from app.main import app


@pytest.fixture(scope="session")
def mock_classification_image(tmpdir_factory):
    url = "https://m.media-amazon.com/images/I/517Nh08xqkL._AC_SX425_.jpg"
    return requests.get(url).content


@pytest_asyncio.fixture(scope="function")
async def test_app_asyncio():
    # for httpx>=20, follow_redirects=True (cf. https://github.com/encode/httpx/releases/tag/0.20.0)
    async with AsyncClient(app=app, base_url="http://test", follow_redirects=True) as ac:
        yield ac  # testing happens here
