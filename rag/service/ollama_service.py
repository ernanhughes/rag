import re
from typing import List

import ollama


PROMPT = """
<|im_start|>
As a world class transcript summarizer, create a bullet point summary of the
transcript provided.

First include a suitable title for the summary based on the title within the <TITLE> delimiter.
Then include the bullet point summary of the text within the <TEXT> delimiter.

The format of your response needs to be in markdown formatting. Use "- " for bullet points.

######

<TITLE>
{title}

<TEXT>
{chunk}
<|im_end|>
"""


class OllamaService:
    def __init__(
        self, model: str = "llama3.1", window_size: int = 128000, overlap: int = 1000
    ):
        self.model = model
        self.window_size = window_size
        self.overlap = overlap

    def chat_with_model(self, model: str, messages):
        response = ollama.chat(model, messages)
        print(response)
        return response

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Removes extra whitespaces from the input text.

        :param text: a string containing the text to be cleaned.
        :return: a cleaned version of the input text.
        """
        # Remove extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()

    def sliding_window_chunking(self, text: str) -> List[str]:
        """
        Splits the input text into chunks using the sliding window technique.

        :param text: a string containing the text to be chunked.
        :return: a list of chunks generated from the input text.
        """
        text = self.clean_text(text)
        tokens = text.split()
        # If the text contains fewer tokens than window_size, return the text as a single chunk.
        if len(tokens) < self.window_size:
            return [text]

        # Use a list comprehension to create chunks from windows
        step = self.window_size - self.overlap
        # Ensure the range covers the entire length of the tokens
        chunks = [
            " ".join(tokens[i : i + self.window_size])
            for i in range(0, len(tokens) - self.window_size + step, step)
        ]
        return chunks

    @staticmethod
    def pull_model(name: str = "llama3.1"):
        ollama.pull(name)

    @staticmethod
    def create_embedding(model: str, text: str):
        return ollama.embed(model, text)
