"""
summary.py

Automatically create summaries for news items from an LLM
"""

import sqlite3
import time
from typing import Dict, List

import torch
from langchain.globals import set_debug
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_mistralai import ChatMistralAI
from pydantic import Field
from sentence_transformers import SentenceTransformer

from taranis_ds.config import Config
from taranis_ds.llm_tools import create_chain, prompt_model_with_retry
from taranis_ds.log import get_logger
from taranis_ds.misc import check_config, convert_language, detect_language
from taranis_ds.persist import check_column_exists, get_db_connection, insert_column, run_query, update_row


logger = get_logger(__name__)

SUMMARY_PROMPT_TEMPLATE = (
    "Please write me a summary of the following text:\n{text}. Your response must be in {language}. \n"
    "Your response must not be longer than {max_words} words. "
    "Your response must contain only the summary, no other words, headings or tags."
)


class SummaryParser(BaseOutputParser):
    desired_lang: str = Field(default="", description="Desired summary language")
    max_words: int = Field(default=None, description="Desired summary length in words")

    def parse(self, text: str):
        summary_language = detect_language(text)
        summary_word_count = len(text.split(" "))

        if summary_language == "err" or summary_language != self.desired_lang:
            raise OutputParserException("The summary is in the wrong language.")

        if summary_word_count > self.max_words * 1.5:
            raise OutputParserException("The summary is too long.")

        if summary_word_count < self.max_words * 0.5:
            raise OutputParserException("The summary is too short.")

        return text


def assess_summary_quality(original_text: str, summary_text: str) -> float:
    # assess the quality of the summary by calculating the similarity of its embeddings
    # with the embeddings of the full text

    embedding_model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    ref_embedding = torch.tensor(embedding_model.encode(original_text)).squeeze(0)
    summary_embedding = torch.tensor(embedding_model.encode(summary_text)).squeeze(0)
    return torch.nn.CosineSimilarity(dim=0)(ref_embedding, summary_embedding).item()


def create_summaries_for_news_items(
    chat_model: BaseChatModel,
    news_items: List[Dict],
    connection: sqlite3.Connection,
    max_length: int,
    quality_threshold: float,
    min_wait: float,
    debug: bool = False,
):
    if debug:
        set_debug(True)

    summary_parser = SummaryParser(max_words=max_length)
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=summary_parser, llm=chat_model, max_retries=3)

    prompt = PromptTemplate(
        template=SUMMARY_PROMPT_TEMPLATE, input_variables=["text", "language"], partial_variables={"max_words": max_length}
    )

    attempt = 0
    cooldown_count = 0

    for i, row in enumerate(news_items):
        logger.info("Creating summary for news item %s/%s", i + 1, len(news_items))
        prompt_lang = convert_language(row["language"])
        retry_parser.parser.desired_lang = row["language"]
        chain = create_chain(chat_model, prompt, retry_parser)

        summary, status = prompt_model_with_retry(chain, {"text": row["content"], "language": prompt_lang})

        if status == "TOO_MANY_REQUESTS":
            logger.error("Got TOO_MANY_REQUESTS response. Continuing to next item and increasing the wait time.")
            attempt += 1
            cooldown_count = 0

        if summary and assess_summary_quality(row["content"], summary) < quality_threshold:
            status = "LOW_QUALITY"

        logger.info("STATUS: %s", status)
        try:
            update_row(connection, "results", row["id"], ["summary", "summary_status"], [summary, status])
        except RuntimeError as e:
            logger.error(e)

        sleep_time = min(10.0, max(min_wait, min_wait * (2**attempt)))
        logger.debug("Waiting %s s before next request", sleep_time)
        time.sleep(sleep_time)
        cooldown_count += 1
        if cooldown_count == 5:
            cooldown_count = 0
            if attempt > 0:
                attempt -= 1

    set_debug(False)


def run():
    logger.info("Running summary step")
    for conf_name, conf_type in [("SUMMARY_MODEL", str), ("SUMMARY_API_KEY", str), ("SUMMARY_ENDPOINT", str), ("SUMMARY_MAX_LENGTH", int)]:
        if not check_config(conf_name, conf_type):
            logger.error("Skipping summary step")
            return

    connection = get_db_connection(Config.DB_PATH, init=True)

    for col in ["summary", "summary_status"]:
        if not check_column_exists(connection, "results", col):
            insert_column(connection, "results", col, "TEXT")
            return
    try:
        query_result = run_query(
            connection,
            "SELECT id, content, language FROM results WHERE summary_status IS NOT 'OK'",
        )
    except RuntimeError as e:
        logger.error(e)
        return
    logger.info("Creating summaries for %s news items", len(query_result))
    news_items = [{"id": row[0], "content": row[1], "language": row[2]} for row in query_result]

    chat_model = ChatMistralAI(
        model=Config.SUMMARY_MODEL,
        api_key=Config.SUMMARY_API_KEY,
        endpoint=Config.SUMMARY_ENDPOINT,
        max_tokens=Config.SUMMARY_MAX_LENGTH * 2,
    )

    create_summaries_for_news_items(
        chat_model,
        news_items,
        connection,
        Config.SUMMARY_MAX_LENGTH,
        Config.SUMMARY_QUALITY_THRESHOLD,
        Config.SUMMARY_REQUEST_WAIT_TIME,
        Config.DEBUG,
    )


if __name__ == "__main__":
    run()
