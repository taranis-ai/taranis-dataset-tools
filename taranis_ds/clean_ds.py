"""
clean_ds.py
"""

import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer


def detect_lang(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "err"


def df_iterator(df: pd.DataFrame, bs: int):
    # yield batches of size bs from df

    for i in range(0, len(df), bs):
        yield df.iloc[i : i + bs]


def get_tokens(df: pd.DataFrame, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_lens = []

    text_iter = df_iterator(df, 32)

    for batch in text_iter:
        texts = batch["content"].to_list()
        tokens = tokenizer(texts)["input_ids"]
        token_lens.extend(list(map(len, tokens)))
    return token_lens


def clean_taranis_dataset(ds_path: str, tokenizer_name: str, max_tokens: int | None = None) -> pd.DataFrame:
    df = pd.read_json(ds_path)

    # remove NoneType or empty News items
    df = df[~df["news_items"].apply(lambda item: item[0]["content"] is None or item[0]["content"] == "")]

    df["content"] = df["news_items"].apply(lambda item: item[0]["content"])
    df = df[~df["content"].duplicated()]

    df["title"] = df["news_items"].apply(lambda item: item[0]["title"])
    df["news_item_id"] = df["news_items"].apply(lambda item: item[0]["id"])

    df["tokens"] = get_tokens(df, tokenizer_name)
    df["language"] = df["content"].apply(detect_lang)
    df = df[~df["language"] == "err"]

    if max_tokens is not None:
        df = df[df["tokens"] <= max_tokens]

    return df[["id", "news_item_id", "title", "content", "tokens", "language"]]
