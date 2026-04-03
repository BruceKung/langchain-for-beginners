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

    books = [
        Document(
            page_content="A young wizard discovers his magical powers and battles dark forces at a school of magic",
            metadata={"title": "Harry Potter", "author": "J.K. Rowling", "genre": "Fantasy"}
        ),
        Document(
            page_content="A hobbit embarks on an epic quest through Middle-earth to destroy a powerful ring",
            metadata={"title": "The Lord of the Rings", "author": "J.R.R. Tolkien", "genre": "Fantasy"}
        ),
        Document(
            page_content="A young girl discovers she has the ability to fly and uses it to help others",
            metadata={"title": "The Secret Life of Bees", "author": "Sue Monk Kidd", "genre": "Drama"}
        ),
        Document(
            page_content="A young boy discovers he has the ability to time travel and uses it to help others",
            metadata={"title": "The Time Traveler's Wife", "author": "Audrey Niffenegger", "genre": "Romance"}
        ),
    ]

    vector_store = InMemoryVectorStore.from_documents(books, embeddings)

    queries = [
        "stories about magic and wizards",
        "epic journey adventures",
        "books about the future",
        "mystery and detective stories",
    ]
    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=3)
        for doc, score in results:
            print(f"Score: {score:.4f} - {doc.page_content}")
            print(f"Title: {doc.metadata['title']}")
            print(f"Author: {doc.metadata['author']}")
            print(f"Genre: {doc.metadata['genre']}")
            print("-" * 80)
        print("-" * 80)

if __name__ == "__main__":
    main()