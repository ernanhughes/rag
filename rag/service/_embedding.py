import ollama
from rag._config import appConfig


class EmbeddingService:
    def __init__(self, model_name: str = appConfig.get("OLLAMA_EMBEDDING_MODEL")):
        self.model_name = model_name

    def embed(self, text: str):
        return ollama.embeddings(model=self.model_name, prompt=text)["embedding"]

