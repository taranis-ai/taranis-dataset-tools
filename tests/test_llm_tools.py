import httpx
from taranis_ds import llm_tools
from unittest.mock import patch, MagicMock


@patch("taranis_ds.llm_tools.create_chain")
def test_prompt_model_with_retry(mock_create_chain):

    mock_chain = MagicMock()
    mock_create_chain.return_value = mock_chain

    # 200 response
    mock_chain.invoke.return_value = "Output"
    output, status = llm_tools.prompt_model_with_retry(mock_chain, {}, 3)
    assert output == "Output"
    assert status == "OK"

    # 500 response
    server_error_response = httpx.HTTPError("500 Internal Server Error")
    mock_chain.invoke.side_effect = server_error_response
    output, status = llm_tools.prompt_model_with_retry(mock_chain, {}, 3)
    assert output == ""
    assert status == "ERROR"

    # 429 response
    too_many_requests_response = httpx.HTTPError("429 Too Many Requests")
    mock_chain.invoke.side_effect = too_many_requests_response
    output, status = llm_tools.prompt_model_with_retry(mock_chain, {}, 3)
    assert mock_chain.invoke.call_count == 5
    assert output == ""
    assert status == "TOO_MANY_REQUESTS"
