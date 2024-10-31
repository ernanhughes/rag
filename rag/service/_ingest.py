import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag._models import Models
from rag._database import RagDb
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

class IngestService:
    def __init__(self):
        self.rag_db = RagDb()
        # Initialize the models
        self.models = Models()
        self.embeddings = self.models.embeddings_ollama
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

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000, chunk_overlap=50, separators=["\n", " ", ""]
        # )
        # documents = text_splitter.split_documents(document_text)
        # uuids = [str(uuid4()) for _ in range(len(documents))]
        # print(f"Adding {len(documents)} documents to the vector store")
        # print(f"Finished ingesting file: {file_path}")


    def ingest_folder(self, data_folder: str):  # Main loop
        for filename in os.listdir(data_folder):
            if not filename.startswith("_"):
                file_path = os.path.join(data_folder, filename)
                self.ingest_file(file_path)
                # new_filename = "_" + filename
                # new_file_path = os.path.join(data_folder, new_filename)
                # os.rename(file_path, new_file_path)

