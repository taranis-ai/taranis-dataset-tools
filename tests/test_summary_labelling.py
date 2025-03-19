import httpx
from taranis_ds import summary
import taranis_ds.summary
from unittest.mock import patch, MagicMock, Mock
from langchain.chat_models.base import BaseChatModel
from .testdata import REF_NEWS_ITEM_DE, REF_NEWS_ITEM_EN, REF_SUMMARY_DE, REF_SUMMARY_EN, NON_SUMMARY_DE



def test_convert_language():
    assert summary.convert_language("fr") == "french"
    assert summary.convert_language("de") == "german"
    assert summary.convert_language("en") == "english"
    assert summary.convert_language("ru") == "russian"


@patch("taranis_ds.summary.time.sleep")
@patch("taranis_ds.summary.create_chain")
def test_create_summaries_for_news_items(mock_create_chain, mock_sleep, results_db):

    chat_model = Mock(spec=BaseChatModel)
    mock_chain = MagicMock()
    mock_create_chain.return_value = mock_chain

    # successful summary creation
    news_items = [{"id": "1", "content": REF_NEWS_ITEM_DE, "language": "de"}]
    mock_chain.invoke.return_value = REF_SUMMARY_DE
    summary.create_summaries_for_news_items(chat_model, news_items, results_db, "results", 300, 0.5, 1.)
    saved_results = results_db.execute("SELECT summary, summary_status FROM results WHERE id='1'").fetchall()
    assert saved_results == [(REF_SUMMARY_DE, "OK")]

    # 500 response
    server_error_response = httpx.HTTPError("500 Internal Server Error")
    mock_chain.invoke.side_effect = server_error_response
    summary.create_summaries_for_news_items(chat_model, news_items, results_db, "results", 300, 0.5, 1.)
    saved_results = results_db.execute("SELECT summary, summary_status FROM results WHERE id='1'").fetchall()
    assert saved_results == [('', "ERROR")]

    # bad summary
    mock_chain.invoke.side_effect = None
    mock_chain.invoke.return_value = NON_SUMMARY_DE
    summary.create_summaries_for_news_items(chat_model, news_items, results_db, "results", 300, 0.8, 1.)
    saved_results = results_db.execute("SELECT summary, summary_status FROM results WHERE id='1'").fetchall()
    assert saved_results == [(NON_SUMMARY_DE, "LOW_QUALITY")]

    # 429 response
    too_many_requests_response = httpx.HTTPError("429 Too Many Requests")
    mock_chain.invoke.side_effect = too_many_requests_response
    summary.create_summaries_for_news_items(chat_model, news_items, results_db, "results", 300, 0.5, 1.)
    assert mock_chain.invoke.call_count == 6
    saved_results = results_db.execute("SELECT summary, summary_status FROM results WHERE id='1'").fetchall()
    assert saved_results == [('', "TOO_MANY_REQUESTS")]

