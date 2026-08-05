"""Tests for utils/preprocessing.py.

Uses only small in-memory data; never touches the real WELFake dataset.
"""

import pandas as pd
import pytest

from utils.preprocessing import TextPreprocessor


@pytest.fixture
def preprocessor():
    return TextPreprocessor()


class TestCleanText:
    def test_normal_text(self, preprocessor):
        assert preprocessor.clean_text("Hello World") == "hello world"

    def test_empty_string(self, preprocessor):
        assert preprocessor.clean_text("") == ""

    def test_none_input(self, preprocessor):
        assert preprocessor.clean_text(None) == ""

    def test_numeric_input(self, preprocessor):
        assert preprocessor.clean_text(12345) == ""

    def test_url_removal(self, preprocessor):
        result = preprocessor.clean_text("Visit https://example.com for more info")
        assert "http" not in result
        assert "example.com" not in result

    def test_www_url_removal(self, preprocessor):
        result = preprocessor.clean_text("Go to www.example.com now")
        assert "example.com" not in result

    def test_html_removal(self, preprocessor):
        result = preprocessor.clean_text("<b>Breaking</b> <i>news</i>")
        assert "<" not in result
        assert ">" not in result
        assert "breaking" in result
        assert "news" in result

    def test_lowercase_conversion(self, preprocessor):
        result = preprocessor.clean_text("SHOUTING Headline")
        assert result == result.lower()
        assert "shouting" in result

    def test_punctuation_removal(self, preprocessor):
        result = preprocessor.clean_text("Wait... really?! Yes, absolutely.")
        for char in ".!?,":
            assert char not in result

    def test_number_removal(self, preprocessor):
        result = preprocessor.clean_text("Article 12345 published in 2024")
        assert not any(ch.isdigit() for ch in result)

    def test_repeated_whitespace_collapsed(self, preprocessor):
        result = preprocessor.clean_text("Too    many     spaces")
        assert "  " not in result
        assert result == "too many spaces"


class TestPreprocessDataframe:
    def test_creates_content_column(self, preprocessor):
        df = pd.DataFrame({"title": ["Hello"], "text": ["World"]})

        result = preprocessor.preprocess_dataframe(df)

        assert "content" in result.columns
        assert result.loc[0, "content"] == "hello world"

    def test_handles_missing_values(self, preprocessor):
        df = pd.DataFrame({"title": [None], "text": ["Some text"]})

        result = preprocessor.preprocess_dataframe(df)

        assert result.loc[0, "content"] == "some text"

    def test_missing_title_column_raises(self, preprocessor):
        df = pd.DataFrame({"text": ["Some text"]})

        with pytest.raises(ValueError):
            preprocessor.preprocess_dataframe(df)

    def test_missing_text_column_raises(self, preprocessor):
        df = pd.DataFrame({"title": ["Some title"]})

        with pytest.raises(ValueError):
            preprocessor.preprocess_dataframe(df)

    def test_label_column_preserved(self, preprocessor):
        df = pd.DataFrame({"title": ["Hello"], "text": ["World"], "label": [1]})

        result = preprocessor.preprocess_dataframe(df)

        assert result.loc[0, "label"] == 1

    def test_full_dataframe_round_trip(self, preprocessor, sample_dataframe):
        result = preprocessor.preprocess_dataframe(sample_dataframe)

        assert len(result) == len(sample_dataframe)
        assert "content" in result.columns
        # Original labels must be untouched by preprocessing.
        assert result["label"].tolist() == sample_dataframe["label"].tolist()
        # Rows with a missing title/text must not contain "none" text.
        assert "none" not in result.loc[1, "content"]
