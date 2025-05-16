from sentence_transformers import SentenceTransformer
import os

def load_documents(file_path=None):
    if file_path is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "..", "static_files", "knowledge_base.txt")

    with open(file_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


documents = load_documents()
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
