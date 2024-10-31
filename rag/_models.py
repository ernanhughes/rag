from langchain_ollama import OllamaEmbeddings, ChatOllama
from rag._config import appConfig
import ollama


class Models:
    def __init__(
        self,
        ollama_url: str = appConfig.get("OLLAMA_URL"),
        ollama_model: str = appConfig.get("OLLAMA_MODEL"),
        ollama_embedding_model: str = appConfig.get("OLLAMA_EMBEDDING_MODEL"),
    ):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.ollama_embedding_model = ollama_embedding_model
        self.embeddings_ollama = OllamaEmbeddings(model=self.ollama_embedding_model)
        self.model_ollama = ChatOllama(
            model=ollama_model, temperature=0.7
        )

    @staticmethod
    def pull_model(name: str = appConfig.get("OLLAMA_MODEL")):
        ollama.pull(name)

    @staticmethod
    def create_embedding(text: str, model: str = appConfig.get("OLLAMA_EMBEDDING_MODEL")):
        return ollama.embed(model, text)
