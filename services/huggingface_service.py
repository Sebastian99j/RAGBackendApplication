from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from core.config import settings
from services.retriever import get_context

# distilgpt2
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)

def ask_hf(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    prompt = f"Odpowiedz na pytanie: {question}\nNa podstawie:\n{context}\nOdpowiedź:"
    return hf_pipe(prompt)[0]['generated_text'][len(prompt):].strip()

# gpt2
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_pipe = pipeline("text-generation", model=model_gpt2, tokenizer=tokenizer_gpt2, max_new_tokens=100)

def ask_gpt2(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    prompt = f"Odpowiedz na pytanie: {question}\nNa podstawie:\n{context}\nOdpowiedź:"
    return gpt2_pipe(prompt)[0]["generated_text"][len(prompt):].strip()

# API (np. Mistral przez HuggingFace Inference API)
mistral_pipe = pipeline(
    "text-generation",
    model="gpt2",
    token=settings.HF_API_TOKEN,
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
    return mistral_pipe(prompt)[0]['generated_text'].strip()
