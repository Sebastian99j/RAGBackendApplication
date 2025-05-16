from fastapi import APIRouter
from api.schemas import QueryRequest, FineTuneRequest
from services.openai_service import ask_openai
from services.huggingface_service import ask_hf, ask_gpt2, ask_hf_api
from services.langchain_service import ask_langchain, ask_langchain_faiss
from services.fine_tune_service import fine_tune_gpt2


router = APIRouter()


@router.post("/ask-openai")
def endpoint_openai(req: QueryRequest):
    return {"model": "openai", "answer": ask_openai(req.question, req.top_k, req.database)}


@router.post("/ask-hf")
def endpoint_hf(req: QueryRequest):
    return {"model": "huggingface", "answer": ask_hf(req.question, req.top_k, req.database)}


@router.post("/ask-langchain")
def endpoint_langchain(req: QueryRequest):
    return {"model": "langchain", "answer": ask_langchain(req.question, req.top_k, req.database)}


@router.post("/ask-gpt2")
def endpoint_gpt2(req: QueryRequest):
    return {"model": "gpt2-local", "answer": ask_gpt2(req.question, req.top_k, req.database)}


@router.post("/ask-hf-api")
def endpoint_hf_api(req: QueryRequest):
    return {"model": "mistral-api", "answer": ask_hf_api(req.question, req.top_k, req.database)}


@router.post("/langchain-faiss")
def endpoint_langchain_faiss(req: QueryRequest):
    return {"model": "langchain-faiss", "answer": ask_langchain_faiss(req.question)}


@router.post("/fine-tune-gpt2")
def fine_tune(req: FineTuneRequest):
    return fine_tune_gpt2(req)
