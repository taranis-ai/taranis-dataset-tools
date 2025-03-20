from taranis_ds import summary
from unittest.mock import patch, MagicMock, Mock
from langchain.chat_models.base import BaseChatModel
from .testdata import REF_NEWS_ITEM_DE, REF_SUMMARY_DE, NON_SUMMARY_DE



def test_convert_language():
    assert summary.convert_language("fr") == "french"
    assert summary.convert_language("de") == "german"
    assert summary.convert_language("en") == "english"
    assert summary.convert_language("ru") == "russian"


@patch("taranis_ds.summary.prompt_model_with_retry")
def test_create_summaries_for_news_items(mock_llm_response, results_db):

    chat_model = Mock(spec=BaseChatModel)

    # successful summary creation
    news_items = [{"id": "1", "content": REF_NEWS_ITEM_DE, "language": "de"}]
    mock_llm_response.return_value = (REF_SUMMARY_DE, "OK")
    summary.create_summaries_for_news_items(chat_model, news_items, results_db, "results", 300, 0.5, 1.)
    saved_results = results_db.execute("SELECT summary, summary_status FROM results WHERE id='1'").fetchall()
    assert saved_results == [(REF_SUMMARY_DE, "OK")]

    # 500 response
    mock_llm_response.return_value = "", "ERROR"
    summary.create_summaries_for_news_items(chat_model, news_items, results_db, "results", 300, 0.5, 1.)
    saved_results = results_db.execute("SELECT summary, summary_status FROM results WHERE id='1'").fetchall()
    assert saved_results == [('', "ERROR")]

    # bad summary
    mock_llm_response.return_value = NON_SUMMARY_DE, "OK"
    summary.create_summaries_for_news_items(chat_model, news_items, results_db, "results", 300, 0.8, 1.)
    saved_results = results_db.execute("SELECT summary, summary_status FROM results WHERE id='1'").fetchall()
    assert saved_results == [(NON_SUMMARY_DE, "LOW_QUALITY")]

    # 429 response
    mock_llm_response.return_value = "", "TOO_MANY_REQUESTS"
    summary.create_summaries_for_news_items(chat_model, news_items, results_db, "results", 300, 0.5, 1.)
    saved_results = results_db.execute("SELECT summary, summary_status FROM results WHERE id='1'").fetchall()
    assert saved_results == [('', "TOO_MANY_REQUESTS")]

