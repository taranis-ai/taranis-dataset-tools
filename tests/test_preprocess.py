from taranis_ds import preprocess
import pandas as pd

def test_get_tokens():

    df = pd.DataFrame([{"content": "a"},
                       {"content": "The quick brown fox"},
                       {"content": "0 1 2 3"}])

    token_lengths = preprocess.get_tokens(df, "facebook/bart-large-cnn")
    assert token_lengths == [3, 6, 6]
