import faiss
from core.documents import embedder, documents, doc_embeddings
from core.config import settings
from pinecone import Pinecone, ServerlessSpec

dimension = doc_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(doc_embeddings)

pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
index_name = "rag-index"

if index_name not in [idx.name for idx in pinecone.list_indexes()]:
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pinecone.Index(index_name)

pinecone_vectors = [
    {"id": f"doc-{i}", "values": vec.tolist(), "metadata": {"text": documents[i]}}
    for i, vec in enumerate(doc_embeddings)
]
for i in range(0, len(pinecone_vectors), 100):
    pinecone_index.upsert(vectors=pinecone_vectors[i:i + 100])

def get_context(question: str, top_k: int, backend: str):
    query_vec = embedder.encode([question])
    if backend == "pinecone":
        res = pinecone_index.query(vector=query_vec[0].tolist(), top_k=top_k, include_metadata=True)
        return "\n".join([match['metadata']['text'] for match in res.matches])
    else:
        vec = query_vec.astype("float32")
        _, idxs = faiss_index.search(vec, k=top_k)
        return "\n".join([documents[i] for i in idxs[0]])
