"""
cybersecurity_classification.py

Classify news items in Cybersecurity/Non-Cybersecurity
"""

import sqlite3
import time
from typing import Dict, List

from langchain.globals import set_debug
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_mistralai import ChatMistralAI
from llm_tools import create_chain, prompt_model_with_retry

from taranis_ds.config import Config
from taranis_ds.log import get_logger
from taranis_ds.misc import convert_language
from taranis_ds.persist import check_column_exists, check_table_exists, get_db_connection, run_query, update_row


logger = get_logger(__name__)

CATEGORIES = ["Cybersecurity", "Non-Cybersecurity"]

CYBERSEC_CLASS_PROMPT_TEMPLATE = (
    f"Please classify the following text into one of two categories: {CATEGORIES[0]} or {CATEGORIES[1]}.\n"
    "The text is in {language}. \n"
    "Respond only with the category name.\n"
    "Text: {text}"
)


class CategoryOutputParser(BaseOutputParser):
    def parse(self, text: str):
        if text.strip() not in CATEGORIES:
            raise OutputParserException(f"Invalid output: {text}. The output should be only one of {CATEGORIES[0]} or {CATEGORIES[1]}")
        return text


def classify_news_item_cybersecurity(
    chat_model: BaseChatModel,
    news_items: List[Dict],
    connection: sqlite3.Connection,
    table_name: str,
    wait_time: float,
    debug: bool = False,
):
    if debug:
        set_debug(True)

    category_parser = CategoryOutputParser()
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=category_parser, llm=chat_model, max_retries=3)

    prompt = PromptTemplate(template=CYBERSEC_CLASS_PROMPT_TEMPLATE, input_variables=["language", "text"])

    backoff_coeff = 0.0
    cooldown_count = 0.0

    for row in news_items:
        prompt_lang = convert_language(row["language"])
        retry_parser.parser.desired_lang = row["language"]
        chain = create_chain(chat_model, prompt, retry_parser)

        category, status = prompt_model_with_retry(chain, wait_time, backoff_coeff, {"language": prompt_lang, "text": row["content"]})

        if status == "TOO_MANY_REQUESTS":
            backoff_coeff += 1
            cooldown_count = 0

        try:
            update_row(connection, table_name, row["id"], ["cybersecurity", "cybersecurity_status"], [category, status])
        except RuntimeError as e:
            logger.error(e)

        time.sleep(wait_time**backoff_coeff)
        cooldown_count += 1
        if cooldown_count == 5:
            cooldown_count = 0
            if backoff_coeff > 0:
                backoff_coeff -= 1

    set_debug(False)


def run():
    connection = get_db_connection(Config.DB_PATH, init=True)
    if not check_table_exists(connection, Config.TABLE_NAME):
        logger.error("Table %s does not exist. Cannot create summaries", Config.TABLE_NAME)
        return

    for col in ["cybersecurity", "cybersecurity_status"]:
        if not check_column_exists(connection, Config.TABLE_NAME, col):
            logger.error("The column '%s' does not exist in the table %s. Create it first", col, Config.TABLE_NAME)
            return
    try:
        query_result = run_query(connection, f"SELECT id, content, language FROM {Config.TABLE_NAME} WHERE summary_status != 'OK'")
    except RuntimeError as e:
        logger.error(e)
        return
    news_items = [{"id": row[0], "content": row[1], "language": row[2]} for row in query_result]

    chat_model = ChatMistralAI(
        model=Config.SUMMARY_TEACHER_MODEL,
        api_key=Config.SUMMARY_TEACHER_API_KEY,
        endpoint=Config.SUMMARY_TEACHER_ENDPOINT,
        max_tokens=Config.SUMMARY_MAX_TOKENS,
    )

    classify_news_item_cybersecurity(
        chat_model,
        news_items,
        connection,
        Config.TABLE_NAME,
        Config.SUMMARY_REQUEST_WAIT_TIME,
        Config.DEBUG,
    )


if __name__ == "__main__":
    run()
