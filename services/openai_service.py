from openai import OpenAI
from core.config import settings
from services.retriever import get_context
from core.redis_client import get_chat_history, add_message_to_history

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def ask_openai(session_id: str,question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    history = get_chat_history(session_id)

    chat_prompt = ""
    for h in history:
        chat_prompt += f"Użytkownik: {h['user']}\nAsystent: {h['assistant']}\n"
    chat_prompt += f"Użytkownik: {question}\nAsystent:"

    full_prompt = f"Kontekst:\n{context}\n\n{chat_prompt}"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2
    )
    reply = response.choices[0].message.content.strip()

    add_message_to_history(session_id, question, reply)
    return reply
