
from pathlib import Path
import csv

from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore

base = Path(__file__).resolve().parents[2]
data = base / "data"
output = base / "output"
output.mkdir(exist_ok=True)

pdfs = sorted(data.glob("*.pdf"))
if len(pdfs) != 5:
    raise ValueError("Add exactly 5 PDFs to the data folder.")

converter = PyPDFToDocument()
documents = []

for pdf in pdfs:
    for doc in converter.run(sources=[pdf])["documents"]:
        doc.meta["source_file"] = pdf.name
        documents.append(doc)

documents = [d for d in documents if d.content and d.content.strip()]

questions = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "What is deep learning?",
    "How are robots controlled?",
    "What is sensor fusion?",
    "How do autonomous vehicles navigate?",
    "What is computer vision?",
    "How are neural networks trained?",
    "What are embedded systems?",
    "How can anomalies be detected?",
]

bm25_store = InMemoryDocumentStore()
bm25_store.write_documents(documents)

bm25 = Pipeline()
bm25.add_component(
    "retriever",
    InMemoryBM25Retriever(document_store=bm25_store, top_k=3),
)

model = "sentence-transformers/all-MiniLM-L6-v2"

doc_embedder = SentenceTransformersDocumentEmbedder(model=model)
doc_embedder.warm_up()
embedded = doc_embedder.run(documents)["documents"]

dense_store = InMemoryDocumentStore(
    embedding_similarity_function="cosine"
)
dense_store.write_documents(embedded)

query_embedder = SentenceTransformersTextEmbedder(model=model)
query_embedder.warm_up()

dense = Pipeline()
dense.add_component("embedder", query_embedder)
dense.add_component(
    "retriever",
    InMemoryEmbeddingRetriever(document_store=dense_store, top_k=3),
)
dense.connect("embedder.embedding", "retriever.query_embedding")

with (output / "results.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "Method", "Rank", "PDF", "Text", "Relevant"])

    for question in questions:
        for method, pipeline, inputs in [
            ("BM25", bm25, {"retriever": {"query": question}}),
            ("Dense", dense, {"embedder": {"text": question}}),
        ]:
            result = pipeline.run(inputs)
            docs = result["retriever"]["documents"]

            print(f"\n{method}: {question}")

            for rank, doc in enumerate(docs, 1):
                text = " ".join(doc.content.split())[:200]
                print(rank, doc.meta.get("source_file"), text)
                writer.writerow([
                    question, method, rank,
                    doc.meta.get("source_file", ""),
                    text, ""
                ])

print("\nDone. Check output/results.csv")
