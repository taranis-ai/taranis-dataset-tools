"""
llm_tools.py

Common functions for interacting with the LLM
"""

import time

import httpx
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.base import RunnableSequence

from taranis_ds.log import get_logger


logger = get_logger(__name__)


def create_chain(model: BaseChatModel, prompt: PromptTemplate, parser: BaseOutputParser):
    completion_chain = prompt | model | RunnableLambda(lambda x: x.content)
    chain = RunnableParallel(completion=completion_chain, prompt_value=prompt) | RunnableLambda(lambda x: parser.parse_with_prompt(**x))
    return chain


def prompt_model_with_retry(chain: RunnableSequence, model_inputs: dict, max_retries: int = 3) -> tuple[str, str]:
    for _ in range(max_retries):
        try:
            output, status = "", "OK"
            output = chain.invoke(model_inputs)
            break
        except httpx.HTTPError as e:
            status = "ERROR"

            if "429" in str(e):
                time.sleep(0.5)
            else:
                break
        except OutputParserException as e:
            status = "ERROR"
            logger.error("Could not parse LLM output. Error: %s Skipping.", e)

    else:  # got 429 on all tries
        status = "TOO_MANY_REQUESTS"

    return output, status
