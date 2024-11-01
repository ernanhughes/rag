import logging
import os

from langchain_community.document_loaders import PyPDFLoader

from rag._database import RagDb
from rag._models import Models
from rag.split._spacy_text_splitter import SpacyTextSplitter
from rag.service._embedding import EmbeddingService
from sqlite_vec import serialize_float32


logger = logging.getLogger(__name__)


class IngestService:
    def __init__(self):
        self.rag_db = RagDb()
        self.models = Models()
        self.embedder = EmbeddingService(self.models.ollama_embedding_model)
        self.llm = self.models.model_ollama
        self.data_folder = "./data"

    def ingest_file(self, file_path: str):
        # Skip non-PDF files
        if not file_path.lower().endswith('.pdf'):
            print(f"Skipping non-PDF file: {file_path}")
            return

        print(f"Starting to ingest file: {file_path}")
        loader = PyPDFLoader(file_path)
        file_id = self.rag_db.insert_document(file_path)
        logger.info(f"Inserted file {file_path} with id {file_id}")
        documents = loader.load()
        pdf_text = "\n".join([doc.page_content for doc in documents])
        self.rag_db.insert_document_text(file_id, pdf_text)
        logger.info(f"Inserted document text for file {file_path}")

        text_splitter = SpacyTextSplitter()
        for doc in documents:
            chunks = text_splitter.split(doc.page_content)
            logger.info(f"Splitted document into {len(chunks)} chunks")
            for chunk in chunks:
                chunk_id = self.rag_db.insert_document_text(file_id, chunk.text)
                logger.info(f"Inserted document text chunk {chunk_id} for file {file_path}")
                embedding = self.embedder.embed(chunk.text)
                logger.info(f"Created embedding for chunk {chunk_id}")
                self.rag_db.insert_document_embedding(chunk_id, embedding)

  
    def ingest_folder(self, data_folder: str):  
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            if not self.rag_db.contains_document(file_path):
                self.ingest_file(file_path)
