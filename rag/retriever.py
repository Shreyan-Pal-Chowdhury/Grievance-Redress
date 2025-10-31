# rag/retriever.py
from rag.vector_store import RAGVectorStore

class RAGRetriever:
    def __init__(self, json_files: list[str]):
        self.vector_store = RAGVectorStore()
        self.vector_store.build_from_jsons(json_files)
        self.retriever = self.vector_store.get_retriever()

    def retrieve(self, query: str, top_k: int = 3):
        # Returns list of Documents
        docs = self.vector_store.query(query, k=top_k)
        return docs
