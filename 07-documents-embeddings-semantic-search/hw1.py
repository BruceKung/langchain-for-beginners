import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
import requests

load_dotenv()

EMBEDDING_MODEL = "text-embedding-v4"
API_KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"

class AliEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = API_KEY
        self.url = BASE_URL
        self.model = EMBEDDING_MODEL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(self.url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return [item["embedding"] for item in result["data"]]

def main():
    embeddings = AliEmbeddings()

    docs = [
        Document(page_content="Machine learning models can recognize patterns in data"),
        Document(page_content="The recipe calls for flour, eggs, and butter"),
        Document(page_content="Python is a popular programming language for AI"),
        Document(page_content="The sunset painted the sky in shades of orange"),
    ]

    vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

    queries = [
        "Machine learning is able to identify patterns in data",
        "this recipe require butter and eggs",
        "python is good for artificial intelligence",
        "the sunset is beautiful",
    ]

    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=4)
        # print all results rank by similarity score from high to low
        for doc, score in results:
            print(f"Score: {score:.4f} - {doc.page_content}")
        print("-" * 80)

if __name__ == "__main__":
    main()