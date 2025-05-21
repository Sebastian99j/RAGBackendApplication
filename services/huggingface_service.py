from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from core.config import settings
from services.retriever import get_context
from core.redis_client import get_chat_history, add_message_to_history, build_chat_prompt

# distilgpt2
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
hf_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)


def ask_hf(session_id: str, question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    history = get_chat_history(session_id)
    prompt = f"Kontekst:\n{context}\n\n" + build_chat_prompt(history, question)
    output = hf_pipe(prompt)[0]['generated_text']
    reply = output[len(prompt):].strip()
    add_message_to_history(session_id, question, reply)
    return reply

# gpt2
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
model_gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_pipe = pipeline("text-generation", model=model_gpt2, tokenizer=tokenizer_gpt2, max_new_tokens=100)


def ask_gpt2(session_id: str, question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    history = get_chat_history(session_id)
    prompt = f"Kontekst:\n{context}\n\n" + build_chat_prompt(history, question)
    output = gpt2_pipe(prompt)[0]["generated_text"]
    reply = output[len(prompt):].strip()
    add_message_to_history(session_id, question, reply)
    return reply

# API (np. Mistral przez HuggingFace Inference API)
mistral_pipe = pipeline(
    "text-generation",
    model="gpt2",
    token=settings.HF_API_TOKEN,
    pad_token_id=50256,
    max_new_tokens=60,
    return_full_text=False
)


def ask_hf_api(session_id: str, question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    history = get_chat_history(session_id)
    prompt = f"""
    Udziel krótkiej, zwięzłej odpowiedzi na pytanie użytkownika, korzystając z poniższych informacji.

    ### Kontekst:
    {context}

    ### Historia:
    {build_chat_prompt(history, question)}

    ### Odpowiedź:
    """
    reply = mistral_pipe(prompt)[0]['generated_text'].strip()
    add_message_to_history(session_id, question, reply)
    return reply
