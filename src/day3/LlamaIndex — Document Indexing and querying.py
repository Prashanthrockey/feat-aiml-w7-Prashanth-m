
from pathlib import Path
import csv
import time

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


BASE = Path(__file__).resolve().parents[2]
DATA = BASE / "data" / "documents"
OUTPUT = BASE / "output"
OUTPUT.mkdir(exist_ok=True)

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

Settings.embed_model = OllamaEmbedding(
    model_name=EMBED_MODEL
)
Settings.llm = Ollama(
    model=LLM_MODEL,
    request_timeout=120.0,
)

questions = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "What are the types of machine learning?",
    "What is robotics?",
    "What does a robot controller do?",
    "What is sensor fusion?",
    "Which sensors can robots combine?",
    "What is computer vision?",
    "What is object detection?",
    "How does computer vision help autonomous vehicles?",
]

documents = SimpleDirectoryReader(
    input_dir=str(DATA)
).load_data()

if not documents:
    raise ValueError("Add text files to data/documents first.")

print("Documents loaded:", len(documents))

print("\nBuilding normal vector index...")
normal_index = VectorStoreIndex.from_documents(documents)
normal_engine = normal_index.as_query_engine(similarity_top_k=2)

print("\nBuilding ChromaDB index...")
client = chromadb.PersistentClient(
    path=str(BASE / "chroma_db")
)
collection = client.get_or_create_collection("w7d3_documents")

vector_store = ChromaVectorStore(
    chroma_collection=collection
)
chroma_index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
)
chroma_engine = chroma_index.as_query_engine(similarity_top_k=2)

results = []

for question in questions:
    print("\nQuestion:", question)

    for method, engine in [
        ("LlamaIndex", normal_engine),
        ("ChromaDB", chroma_engine),
    ]:
        start = time.perf_counter()
        response = engine.query(question)
        elapsed = time.perf_counter() - start

        answer = str(response)
        print(f"\n{method} ({elapsed:.2f} seconds)")
        print(answer)

        results.append([
            question,
            method,
            round(elapsed, 3),
            answer,
        ])

with (OUTPUT / "query_comparison.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "Question", "Method", "Latency_seconds", "Answer"
    ])
    writer.writerows(results)

print("\nResults saved to output/query_comparison.csv")
