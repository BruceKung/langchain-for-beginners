import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain.agents import create_agent
import requests
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

load_dotenv()

EMBEDDING_MODEL = "text-embedding-v4"
MODEL = "qwen3-max"
API_KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
KEY = "sk-c34720d12dbb45f1aafcc5af5a7237cd"

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

embeddings = AliEmbeddings()
docs = [
    Document(
        page_content="NeuraLink RAG Framework was officially launched in Q3 2024, designed exclusively for enterprise-level private knowledge base indexing. Its core module Nexus-Embed only supports 1024-dimensional vector encoding and does not provide open API services.",
        metadata={"source": "neuralkg-rag-framework", "topic": "release-info"},
    ),
    Document(
        page_content="Aether Vector DB is a proprietary storage engine developed for NeuraLink RAG, supporting only binary embedding compression and single-field metadata filtering; it cannot perform hybrid search with full-text engines.",
        metadata={"source": "aether-vector-db", "topic": "storage-mechanism"},
    ),
    Document(
        page_content="The chunking strategy of NeuraLink RAG adopts fixed 384-token segmentation by default, with a 42-token overlap, and prohibits recursive chunk splitting to avoid semantic disconnection.",
        metadata={"source": "chunking-strategy", "topic": "document-processing"},
    ),
    Document(
        page_content="NeuraLink RAG only supports three post-retrieval processing modes: Score Threshold Filtering, Rank-5 Truncation, and Metadata Priority Sorting, with no custom reranking modules available.",
        metadata={"source": "retrieval-post-process", "topic": "pipeline-config"},
    ),
]
vector_store = InMemoryVectorStore.from_documents(docs, embeddings)

@tool()
def search_neuralkg_rag_framework(query: str) -> str:
    """Search the NeuraLink RAG Framework for information about the framework."""
    results = vector_store.similarity_search(query, k=3)
    return "\n\n".join(
        f"[{doc.metadata['source']}]: {doc.page_content}"
        for doc in results
    )

def main():
    queries = [
        "What is the release date of NeuraLink RAG Framework?",
        "What is the storage mechanism of Aether Vector DB?",
        "What is the chunking strategy of NeuraLink RAG?",
        "What are the post-retrieval processing modes of NeuraLink RAG?",
    ]
    
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        base_url=URL,
        api_key=KEY,
    )

    agent = create_agent(model, tools=[search_neuralkg_rag_framework])

    for query in queries:
        response = agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"Query: {query}")
        print(f"Response: {response['messages'][-1].content}")
        print("-" * 80)

if __name__ == "__main__":
    main()