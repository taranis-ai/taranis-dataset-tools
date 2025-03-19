"""
summary.py

Automatically create summaries for news items from an LLM
"""

import sqlite3
import time
from typing import Dict, List

import httpx
import torch
from iso639 import Lang
from iso639.exceptions import InvalidLanguageValue
from langchain.globals import set_debug
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_mistralai import ChatMistralAI
from pydantic import Field
from sentence_transformers import SentenceTransformer

from taranis_ds.config import Config
from taranis_ds.log import get_logger
from taranis_ds.misc import detect_lang
from taranis_ds.persist import check_column_exists, check_table_exists, get_db_connection, run_query, update_row


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
        summary_language = detect_lang(text)
        summary_word_count = len(text.split(" "))

        if summary_language == "err" or summary_language != self.desired_lang:
            raise OutputParserException("The summary is in the wrong language.")

        if summary_word_count > self.max_words * 1.5:
            raise OutputParserException("The summary is too long.")

        if summary_word_count < self.max_words * 0.5:
            raise OutputParserException("The summary is too short.")

        return text


def convert_language(lang_code: str) -> str:
    try:
        language = Lang(lang_code).name.lower()
    except InvalidLanguageValue:
        return "english"

    return language


def assess_summary_quality(original_text: str, summary_text: str) -> float:
    # assess the quality of the summary by calculating the similarity of its embeddings
    # with the embeddings of the full text

    embedding_model = SentenceTransformer("sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens")
    ref_embedding = torch.tensor(embedding_model.encode(original_text)).squeeze(0)
    summary_embedding = torch.tensor(embedding_model.encode(summary_text)).squeeze(0)
    similarity = torch.nn.CosineSimilarity(dim=0)(ref_embedding, summary_embedding).item()
    return similarity


def create_chain(model, prompt, parser):
    completion_chain = prompt | model | RunnableLambda(lambda x: x.content)
    chain = RunnableParallel(completion=completion_chain, prompt_value=prompt) | RunnableLambda(lambda x: parser.parse_with_prompt(**x))
    return chain


def create_summaries_for_news_items(
    chat_model: BaseChatModel,
    news_items: List[Dict],
    connection: sqlite3.Connection,
    table_name: str,
    max_length: int,
    quality_threshold: float,
    wait_time: float,
    debug: bool = False,
):
    if debug:
        set_debug(True)

    summary_parser = SummaryParser(max_words=max_length)
    retry_parser = RetryWithErrorOutputParser.from_llm(parser=summary_parser, llm=chat_model, max_retries=3)

    prompt = PromptTemplate(
        template=SUMMARY_PROMPT_TEMPLATE, input_variables=["text", "language"], partial_variables={"max_words": max_length}
    )

    backoff_coeff = 0.0
    cooldown_count = 0.0

    for row in news_items:
        prompt_lang = convert_language(row["language"])
        retry_parser.parser.desired_lang = row["language"]
        chain = create_chain(chat_model, prompt, retry_parser)

        for _ in range(3):
            try:
                summary, status = "", "OK"
                summary = chain.invoke({"text": row["content"], "language": prompt_lang})
                break
            except httpx.HTTPError as e:
                status = "ERROR"

                if "429" in str(e):
                    time.sleep(wait_time**backoff_coeff)
                else:
                    break
        else:  # got 429 on all tries
            status = "TOO_MANY_REQUESTS"
            backoff_coeff += 1
            cooldown_count = 0

        if summary:
            if assess_summary_quality(row["content"], summary) < quality_threshold:
                status = "LOW_QUALITY"

        try:
            update_row(connection, table_name, row["id"], ["summary", "summary_status"], [summary, status])
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

    for col in ["summary", "summary_status"]:
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

    create_summaries_for_news_items(
        chat_model,
        news_items,
        connection,
        Config.TABLE_NAME,
        Config.SUMMARY_MAX_LENGTH,
        Config.SUMMARY_QUALITY_THRESHOLD,
        Config.SUMMARY_REQUEST_WAIT_TIME,
        Config.DEBUG,
    )


if __name__ == "__main__":
    run()
