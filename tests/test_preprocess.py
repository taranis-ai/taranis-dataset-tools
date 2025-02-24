from taranis_ds import preprocess
import pandas as pd

def test_get_tokens(tokenizer):

    df = pd.DataFrame([{"content": "a"},
                       {"content": "The quick brown fox"},
                       {"content": "0 1 2 3"}])

    token_lengths = preprocess.get_tokens(df, tokenizer)
    assert token_lengths == [3, 6, 6]

def test_preprocess_taranis_dataset(taranis_dataset_path, tokenizer):

    df = preprocess.preprocess_taranis_dataset(taranis_dataset_path, tokenizer, 1e5)
    assert len(df) == 5
    assert list(df.columns) == ["id", "news_item_id", "title", "content", "tokens", "language"]

    df = preprocess.preprocess_taranis_dataset(taranis_dataset_path, tokenizer, 500)
    assert df["tokens"].max() <= 500
    assert set(df["language"].to_list()) == {"en"}
