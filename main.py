import os
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# === FastAPI app ===
app = FastAPI(title="Multi-LLM RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lub ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Dane wejściowe ===
class QueryRequest(BaseModel):
    question: str
    top_k: int = 2

# === Dokumenty bazowe ===
documents = [
    "Python to język programowania.",
    "FAISS pozwala na szybkie wyszukiwanie wektorowe.",
    "LangChain integruje LLM z bazami danych.",
    "Hugging Face to biblioteka modeli NLP.",
    "GPT to model językowy stworzony przez OpenAI."
]

# === Wektoryzacja ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# === OpenAI GPT ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_openai(question: str, top_k: int):
    query_vec = embedder.encode([question], convert_to_numpy=True)
    _, idxs = index.search(query_vec, k=top_k)
    context = "\n".join([documents[i] for i in idxs[0]])

    prompt = f"""Odpowiedz na podstawie dokumentów:
{context}

Pytanie: {question}
Odpowiedź:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# === Hugging Face GPT2 ===
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

def ask_hf(question: str, top_k: int):
    query_vec = embedder.encode([question], convert_to_numpy=True)
    _, idxs = index.search(query_vec, k=top_k)
    context = "\n".join([documents[i] for i in idxs[0]])

    prompt = f"Odpowiedz na pytanie: {question}\nNa podstawie:\n{context}\nOdpowiedź:"
    output = hf_pipe(prompt)[0]['generated_text']
    return output[len(prompt):].strip()

# === LangChain (nowy styl) ===
prompt = PromptTemplate.from_template("Kontekst: {context}\nPytanie: {question}\nOdpowiedź:")
llm = HuggingFacePipeline(pipeline=hf_pipe)
chain = prompt | llm  # runnable pipeline

def ask_langchain(question: str, top_k: int):
    query_vec = embedder.encode([question], convert_to_numpy=True)
    _, idxs = index.search(query_vec, k=top_k)
    context = "\n".join([documents[i] for i in idxs[0]])
    return chain.invoke({"context": context, "question": question})

# Załaduj GPT-2 (lub distilgpt2 – mniejszy)
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Pipeline do generowania tekstu
gpt2_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

def ask_gpt2(question: str, top_k: int):
    # 1. Wektoryzacja zapytania
    query_vec = embedder.encode([question], convert_to_numpy=True)
    _, idxs = index.search(query_vec, k=top_k)
    context = "\n".join([documents[i] for i in idxs[0]])

    # 2. Zbudowanie prompta
    prompt = f"Odpowiedz na pytanie: {question}\nNa podstawie:\n{context}\nOdpowiedź:"

    # 3. Generowanie tekstu
    output = gpt2_pipe(prompt)[0]["generated_text"]

    # 4. Wyciągnięcie samej odpowiedzi
    return output[len(prompt):].strip()

# microsoft/phi-2
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
mistral_pipe = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    token=HF_API_TOKEN,
    pad_token_id=50256,
    max_new_tokens=60,
    return_full_text=False
)

def ask_hf_api(question: str, top_k: int):
    query_vec = embedder.encode([question], convert_to_numpy=True)
    _, idxs = index.search(query_vec, k=top_k)
    context = "\n".join([documents[i] for i in idxs[0]])

    prompt = f"""
    Udziel krótkiej, zwięzłej odpowiedzi na pytanie użytkownika, korzystając z poniższych informacji.

    ### Kontekst:
    {context}

    ### Pytanie:
    {question}

    ### Odpowiedź:
    """

    output = mistral_pipe(prompt)[0]['generated_text']
    return output.strip()

# === Endpointy API ===
@app.post("/ask-openai")
def endpoint_openai(req: QueryRequest):
    return {"model": "openai", "answer": ask_openai(req.question, req.top_k)}

@app.post("/ask-hf")
def endpoint_hf(req: QueryRequest):
    return {"model": "huggingface", "answer": ask_hf(req.question, req.top_k)}

@app.post("/ask-langchain")
def endpoint_langchain(req: QueryRequest):
    return {"model": "langchain", "answer": ask_langchain(req.question, req.top_k)}

@app.post("/ask-gpt2")
def endpoint_gpt2(req: QueryRequest):
    return {"model": "gpt2-local", "answer": ask_gpt2(req.question, req.top_k)}

@app.post("/ask-hf-api")
def endpoint_hf_api(req: QueryRequest):
    return {"model": "mistral-api", "answer": ask_hf_api(req.question, req.top_k)}
