# rag/vector_store.py
import os
import json
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGVectorStore:
    def __init__(self, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.store = None

    def build_from_jsons(self, json_files: list[str]):
        all_docs = []

        for file_path in json_files:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                if isinstance(item, dict):
                    # Flatten nested JSON to extract text
                    text_parts = []

                    def flatten(d):
                        for k, v in d.items():
                            if isinstance(v, dict):
                                flatten(v)
                            elif isinstance(v, list):
                                for elem in v:
                                    if isinstance(elem, dict):
                                        flatten(elem)
                                    else:
                                        text_parts.append(str(elem))
                            else:
                                text_parts.append(str(v))

                    flatten(item)
                    text = " ".join(text_parts).strip()
                    if text:
                        doc = Document(page_content=text, metadata={"source": file_path})
                        all_docs.append(doc)

        if not all_docs:
            raise ValueError("No documents found in the provided JSON files.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(all_docs)

        self.store = FAISS.from_documents(split_docs, self.embeddings)

    def query(self, query_text: str, k: int = 5):
        if self.store is None:
            raise RuntimeError("Vector store not built. Call build_from_jsons first.")
        return self.store.similarity_search(query_text, k=k)

    def get_retriever(self, **kwargs):
        if self.store is None:
            raise RuntimeError("Vector store not built. Call build_from_jsons first.")
        return self.store.as_retriever(**kwargs)
