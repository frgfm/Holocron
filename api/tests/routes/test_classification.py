import pytest


@pytest.mark.asyncio
async def test_classification(test_app_asyncio, mock_classification_image):

    response = await test_app_asyncio.post("/classification", files={"file": mock_classification_image})
    assert response.status_code == 200
    json_response = response.json()

    # Check that IoU with GT if reasonable
    assert isinstance(json_response, dict)
    assert json_response["value"] == "French horn"
    assert json_response["confidence"] >= 0.8
