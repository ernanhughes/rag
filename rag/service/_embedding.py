import requests
import json

from rag._config import appConfig


class EmbeddingService:
    def __init__(
        self,
        model_name: str = appConfig.get("OLLAMA_EMBEDDING_MODEL"),
        base_url=appConfig.get("OLLAMA_URL"),
    ):
        self.model_name = model_name
        self.base_url = base_url

    def embed(self, text: str):
        return self.generate_embeddings(text=text, 
                                        model=self.model_name,
                                         base_url=self.base_url)["embedding"]

    @staticmethod
    def generate_embeddings(text, model, base_url):
        """Generate embeddings for the given text using the specified model."""
        try:
            # Send a POST request to generate embeddings
            url = f"{base_url}/api/embeddings"
            data = {"prompt": text, "model": model}
            response = requests.post(url, json=data)

            # Check if the request was successful
            if response.status_code == 200:
                # Parse the JSON response
                embeddings = response.json()
                print("Embeddings:")
                pretty_json = json.dumps(embeddings, indent=4)
                print(pretty_json)
                return embeddings
            else:
                print(
                    f"Failed to generate embeddings. Status code: {response.status_code}"
                )
                print("Response:", response.text)
                return None

        except requests.ConnectionError:
            print(
                "Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct."
            )
            return None
        except json.JSONDecodeError:
            print("Failed to parse JSON response from Ollama server.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
