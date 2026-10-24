
import json
import time
import urllib.request
from pathlib import Path
from typing import TypedDict

import chromadb
import mlflow
from langgraph.graph import StateGraph, START, END

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / "data" / "documents"
DB_DIR = BASE / "output" / "chroma_db"
OUTPUT_DIR = BASE / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2:3b"
COLLECTION_NAME = "w7d5_documents"


def ollama_request(endpoint, payload):
    request = urllib.request.Request(
        f"{OLLAMA_URL}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def get_embedding(text):
    result = ollama_request(
        "/api/embed",
        {"model": EMBED_MODEL, "input": text},
    )
    return result["embeddings"][0]


def split_text(text, chunk_size=700, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def build_collection():
    files = list(DATA_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {DATA_DIR}"
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    ids = []
    texts = []
    metadatas = []
    embeddings = []

    for file in files:
        content = file.read_text(encoding="utf-8")

        for number, chunk in enumerate(split_text(content)):
            if not chunk.strip():
                continue

            ids.append(f"{file.stem}_{number}")
            texts.append(chunk)
            metadatas.append({
                "source": file.name,
                "chunk": number,
            })
            embeddings.append(get_embedding(chunk))

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Indexed {len(texts)} chunks from {len(files)} files")
    return collection


class RAGState(TypedDict):
    question: str
    context: str
    sources: list
    answer: str
    latency: float


def retrieve(state):
    start = time.perf_counter()
    vector = get_embedding(state["question"])

    result = collection.query(
        query_embeddings=[vector],
        n_results=3,
        include=["documents", "metadatas"],
    )

    documents = result["documents"][0]
    metadata = result["metadatas"][0]

    context = "\n\n".join(
        f"Source: {meta['source']}\n{text}"
        for text, meta in zip(documents, metadata)
    )

    sources = sorted(set(meta["source"] for meta in metadata))

    return {
        "context": context,
        "sources": sources,
        "latency": time.perf_counter() - start,
    }


def generate(state):
    prompt = (
        "Answer using only the supplied context. "
        "If the context does not contain the answer, say so. "
        "Use simple English.\n\n"
        f"Context:\n{state['context']}\n\n"
        f"Question: {state['question']}"
    )

    result = ollama_request(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a careful document question-answering assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
    )

    return {"answer": result["message"]["content"]}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def main():
    global collection
    collection = build_collection()
    rag = build_graph()

    questions = [
        "What is artificial intelligence?",
        "How does sensor fusion help robots?",
        "What is the role of computer vision in autonomous vehicles?",
    ]

    mlflow.set_tracking_uri(f"sqlite:///{(OUTPUT_DIR / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("W7D5-Multi-Document-RAG")

    results = []

    with mlflow.start_run():
        mlflow.log_param("chat_model", CHAT_MODEL)
        mlflow.log_param("embedding_model", EMBED_MODEL)
        mlflow.log_param("retrieval_top_k", 3)

        for question in questions:
            start = time.perf_counter()

            result = rag.invoke({
                "question": question,
                "context": "",
                "sources": [],
                "answer": "",
                "latency": 0.0,
            })

            total_time = round(time.perf_counter() - start, 2)

            record = {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "response_time_seconds": total_time,
            }
            results.append(record)

            print("\nQuestion:", question)
            print("Answer:", result["answer"])
            print("Sources:", ", ".join(result["sources"]))
            print("Time:", total_time, "seconds")

            mlflow.log_metric(
                "response_time_seconds",
                total_time,
                step=len(results),
            )
            mlflow.log_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                f"question_{len(results)}.json",
            )

    output_file = OUTPUT_DIR / "w7d5_results.json"
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
