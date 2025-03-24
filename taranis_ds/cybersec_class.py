"""
cybersec_class.py

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

from taranis_ds.config import Config
from taranis_ds.llm_tools import create_chain, prompt_model_with_retry
from taranis_ds.log import get_logger
from taranis_ds.misc import check_config, convert_language
from taranis_ds.persist import check_column_exists, get_db_connection, insert_column, run_query, update_row


logger = get_logger(__name__)


CYBERSEC_CLASS_PROMPT_TEMPLATE = (
    "Please classify the following text into one of two categories: 'cybersecurity' or 'non-cybersecurity'\n"
    "The text is in {language}. \n"
    "Respond only with 'cybersecurity' or 'non-cybersecurity'. Do not use any formatting, do not include anything other than one of these words. Do not use quotes.\n"
    "Text: {text}"
)


def process_answer(answer: str):
    answer = answer.strip().strip("\n").strip("\t").strip(".").strip("'").strip('"')
    return answer.lower()


class CategoryOutputParser(BaseOutputParser):
    def parse(self, text: str):
        answer = process_answer(text)
        if answer not in ["cybersecurity", "non-cybersecurity"]:
            raise OutputParserException(f"Invalid output: {text}. The output should be only one of 'cybersecurity' or 'non-cybersecurity'")
        return answer


def classify_news_item_cybersecurity(
    chat_model: BaseChatModel,
    news_items: List[Dict],
    connection: sqlite3.Connection,
    min_wait: float,
    debug: bool = False,
):
    if debug:
        set_debug(True)

    category_parser = CategoryOutputParser()
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=category_parser, llm=chat_model, max_retries=3)

    prompt = PromptTemplate(template=CYBERSEC_CLASS_PROMPT_TEMPLATE, input_variables=["language", "text"])

    attempt = 0
    cooldown_count = 0

    for i, row in enumerate(news_items):
        logger.info("Classifying news item %s/%s", i + 1, len(news_items))
        prompt_lang = convert_language(row["language"])
        chain = create_chain(chat_model, prompt, retry_parser)

        category, status = prompt_model_with_retry(chain, {"language": prompt_lang, "text": row["content"]})

        if status == "TOO_MANY_REQUESTS":
            logger.error("Got TOO_MANY_REQUESTS response. Continuing to next item and increasing the wait time.")
            attempt += 1
            cooldown_count = 0

        logger.info("STATUS: %s", status)
        try:
            update_row(connection, "results", row["id"], ["cybersecurity", "cybersecurity_status"], [category, status])
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
    logger.info("Running cybersecurity classification step")
    for conf_name, conf_type in [("CYBERSEC_CLASS_MODEL", str), ("CYBERSEC_CLASS_API_KEY", str), ("CYBERSEC_CLASS_ENDPOINT", str)]:
        if not check_config(conf_name, conf_type):
            logger.error("Skipping cybersecurity classification step")
            return

    connection = get_db_connection(Config.DB_PATH, "results")

    for col in ["cybersecurity", "cybersecurity_status"]:
        if not check_column_exists(connection, "results", col):
            insert_column(connection, "results", col, "TEXT")
            return
    try:
        query_result = run_query(
            connection,
            "SELECT id, content, language FROM results WHERE cybersecurity_status IS NOT 'OK'",
        )
    except RuntimeError as e:
        logger.error(e)
        return
    logger.info("Classifying %s news items into Cybersecurity/Non-Cybersecurity", len(query_result))
    news_items = [{"id": row[0], "content": row[1], "language": row[2]} for row in query_result]

    chat_model = ChatMistralAI(
        model=Config.CYBERSEC_CLASS_MODEL,
        api_key=Config.CYBERSEC_CLASS_API_KEY,
        endpoint=Config.CYBERSEC_CLASS_ENDPOINT,
        max_tokens=10,
    )

    classify_news_item_cybersecurity(
        chat_model,
        news_items,
        connection,
        Config.CYBERSEC_CLASS_MIN_WAIT_TIME,
        Config.DEBUG,
    )


if __name__ == "__main__":
    run()
