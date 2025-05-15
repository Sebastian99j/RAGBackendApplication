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
from pinecone import Pinecone, ServerlessSpec
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from typing import List
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import uuid
from dotenv import load_dotenv


load_dotenv()

app = FastAPI(title="Multi-LLM RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 2
    database: str = "faiss" # 'faiss' or 'pinecone'

class TrainingExample(BaseModel):
    input: str
    output: str

class FineTuneRequest(BaseModel):
    training_data: List[TrainingExample]
    epochs: int = 3

def load_documents(file_path="knowledge_base.txt"):
    with open(file_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

documents = load_documents()

embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
pinecone = Pinecone(api_key=PINECONE_API_KEY)

index_name = "rag-index"
if index_name not in [idx.name for idx in pinecone.list_indexes()]:
    pinecone.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

pinecone_index = pinecone.Index(index_name)

pinecone_vectors = [
    {
        "id": f"doc-{i}",
        "values": vec.tolist(),
        "metadata": {"text": documents[i]}
    }
    for i, vec in enumerate(doc_embeddings)
]
batch_size = 100
for i in range(0, len(pinecone_vectors), batch_size):
    batch = pinecone_vectors[i:i + batch_size]
    pinecone_index.upsert(vectors=batch)


def get_context(question: str, top_k: int, backend: str):
    query_vec = embedder.encode([question])

    if backend == "pinecone":
        res = pinecone_index.query(vector=query_vec[0].tolist(), top_k=top_k, include_metadata=True)
        return "\n".join([match['metadata']['text'] for match in res.matches])
    else:
        vec = query_vec.astype("float32")
        _, idxs = index.search(vec, k=top_k)
        return "\n".join([documents[i] for i in idxs[0]])


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ask_openai(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)

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


tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)


def ask_hf(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    prompt = f"Odpowiedz na pytanie: {question}\nNa podstawie:\n{context}\nOdpowiedź:"
    output = hf_pipe(prompt)[0]['generated_text']
    return output[len(prompt):].strip()


prompt = PromptTemplate.from_template("Kontekst: {context}\nPytanie: {question}\nOdpowiedź:")
llm = HuggingFacePipeline(pipeline=hf_pipe)
chain = prompt | llm

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.from_texts(documents, embedding_model)
retriever_faiss = faiss_index.as_retriever()


def ask_langchain(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    return chain.invoke({"context": context, "question": question})


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

gpt2_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)


def ask_gpt2(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    prompt = f"Odpowiedz na pytanie: {question}\nNa podstawie:\n{context}\nOdpowiedź:"
    output = gpt2_pipe(prompt)[0]["generated_text"]
    return output[len(prompt):].strip()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
mistral_pipe = pipeline(
    "text-generation",
    #model="microsoft/phi-2",
    model="gpt2",
    token=HF_API_TOKEN,
    pad_token_id=50256,
    max_new_tokens=60,
    return_full_text=False
)


def ask_hf_api(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)

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


@app.get("/healthcheck")
def healthcheck():
    return {"status" : "success"}


@app.post("/ask-openai")
def endpoint_openai(req: QueryRequest):
    return {"model": "openai", "answer": ask_openai(req.question, req.top_k,  req.database)}


@app.post("/ask-hf")
def endpoint_hf(req: QueryRequest):
    return {"model": "huggingface", "answer": ask_hf(req.question, req.top_k,  req.database)}


@app.post("/ask-langchain")
def endpoint_langchain(req: QueryRequest):
    return {"model": "langchain", "answer": ask_langchain(req.question, req.top_k,  req.database)}


@app.post("/ask-gpt2")
def endpoint_gpt2(req: QueryRequest):
    return {"model": "gpt2-local", "answer": ask_gpt2(req.question, req.top_k,  req.database)}


@app.post("/ask-hf-api")
def endpoint_hf_api(req: QueryRequest):
    return {"model": "mistral-api", "answer": ask_hf_api(req.question, req.top_k,  req.database)}


@app.post("/langchain-faiss")
def endpoint_langchain_faiss(req: QueryRequest):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever_faiss)
    answer = qa_chain.run(req.question)
    return {"model": "langchain-faiss", "answer": answer}


@app.post("/fine-tune-gpt2")
def fine_tune_gpt2(req: FineTuneRequest):
    texts = [f"User: {ex.input}\nAI: {ex.output}" for ex in req.training_data]
    dataset = Dataset.from_dict({"text": texts})

    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir="./gpt2-temp-output",
        overwrite_output_dir=True,
        num_train_epochs=req.epochs,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_total_limit=1,
        save_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    save_path = f"./gpt2-finetuned-{uuid.uuid4().hex[:8]}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    global gpt2_pipe
    gpt2_pipe = pipeline("text-generation", model=save_path, tokenizer=save_path, max_new_tokens=100)

    return {
        "status": "success",
        "message": f"Model został przetrenowany i zaktualizowany ({save_path})."
    }
