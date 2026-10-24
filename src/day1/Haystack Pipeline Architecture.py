from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pathlib import Path

files = list(Path("data").glob("*.pdf"))

converter = PyPDFToDocument()
documents = []

for file in files:
    documents.extend(converter.run(sources=[file])["documents"])

store = InMemoryDocumentStore()
store.write_documents(documents)

pipeline = Pipeline()
pipeline.add_component(
    "retriever",
    InMemoryBM25Retriever(document_store=store)
)

result = pipeline.run({
    "retriever": {"query": "What is artificial intelligence?"}
})

for doc in result["retriever"]["documents"]:
    print(doc.content[:200])