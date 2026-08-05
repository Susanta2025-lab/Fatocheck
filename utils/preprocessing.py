import re


class TextPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<.*?>", "", text)

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation (strips out anything that isn't alphanumeric or whitespace)
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def preprocess_dataframe(self, df):
        required_columns = ["title", "text"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        df["title"] = df["title"].fillna("")
        df["text"] = df["text"].fillna("")

        df["content"] = df["title"] + " " + df["text"]

        df["content"] = df["content"].apply(self.clean_text)

        return df
