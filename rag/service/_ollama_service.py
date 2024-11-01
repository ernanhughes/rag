import json
import logging
import re
from typing import List

import ollama
import requests

from rag._config import appConfig

logger = logging.getLogger(__name__)


class OllamaService:
    def __init__(
            self, model: str = "llama3.1", window_size: int = 128000, overlap: int = 1000
    ):
        self.model = model
        self.window_size = window_size
        self.overlap = overlap

    def chat_with_model(self, model: str, messages):
        response = ollama.chat(model, messages)
        logger.info(response)
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
            " ".join(tokens[i: i + self.window_size])
            for i in range(0, len(tokens) - self.window_size + step, step)
        ]
        logger.info(chunks)
        return chunks

    @staticmethod
    def pull_model(name: str = "llama3.1"):
        logger.info(f"Pulling model {name}")
        ollama.pull(name)

    @staticmethod
    def create_embedding(model: str, text: str):
        logger.info(f"Creating embedding for {text} with model {model}")
        return ollama.embed(model, text)

    @staticmethod
    def list_models(ollama_url: str = appConfig.get("OLLAMA_URL")):
        """List installed models from the Ollama server."""
        try:
            # Send a GET request to retrieve the list of installed models
            url = f"{ollama_url}/api/tags"
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                models = response.json()
                logger.info("Installed Ollama Models:")
                for model in models['models']:
                    pretty_json = json.dumps(model, indent=4)
                    logger.info(f'{pretty_json}')
                return models
            else:
                logger.error(f"Failed to retrieve models. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return []

        except requests.ConnectionError:
            logger.error(
                "Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
            return []
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from Ollama server.")
            return []
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return []
