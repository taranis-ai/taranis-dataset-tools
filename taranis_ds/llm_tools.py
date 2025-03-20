"""
llm_tools.py

Common functions for interacting with the LLM
"""

import time

import httpx
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables.base import RunnableSequence


def create_chain(model: BaseChatModel, prompt: PromptTemplate, parser: BaseOutputParser):
    completion_chain = prompt | model | RunnableLambda(lambda x: x.content)
    chain = RunnableParallel(completion=completion_chain, prompt_value=prompt) | RunnableLambda(lambda x: parser.parse_with_prompt(**x))
    return chain


def prompt_model_with_retry(chain: RunnableSequence, wait_time: float, model_inputs: dict, max_retries: int = 3):
    for _ in range(max_retries):
        try:
            output, status = "", "OK"
            output = chain.invoke(model_inputs)
            break
        except httpx.HTTPError as e:
            status = "ERROR"

            if "429" in str(e):
                time.sleep(wait_time)
            else:
                break
    else:  # got 429 on all tries
        status = "TOO_MANY_REQUESTS"

    return output, status
