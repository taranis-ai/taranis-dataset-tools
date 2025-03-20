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
    assert len(df) == 6
    assert list(df.columns) == ["id", "news_item_id", "title", "content", "tokens", "language"]
    assert set(df["language"].to_list()) == {"en"}
    assert df.iloc[0]["id"] == "b57978a3-5009-4c6d-82cb-f92693ad39e7"
    assert df.iloc[3]["title"] == "Test News Item"
    assert df.iloc[-1]["news_item_id"] == "242144c8-07e5-4432-bec3-6a01ed18c65e"

    df = preprocess.preprocess_taranis_dataset(taranis_dataset_path, tokenizer, 300)
    assert df["tokens"].max() <= 300


def test_save_df_to_table(test_db):
    df = pd.DataFrame([{"id": 1, "col1": "Some text", "col2": 666},
                       {"id": 2, "col1": "Yet another text", "col2": 90},
                       {"id": 3, "col1": "three", "col2": 22},
                       ])
    assert preprocess.save_df_to_table(df, test_db) == 3
    assert preprocess.save_df_to_table(df, test_db) == 0
