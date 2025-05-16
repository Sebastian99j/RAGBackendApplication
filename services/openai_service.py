from openai import OpenAI
from core.config import settings
from services.retriever import get_context

client = OpenAI(api_key=settings.OPENAI_API_KEY)

def ask_openai(question: str, top_k: int, database: str):
    context = get_context(question, top_k, database)
    prompt = f"Odpowiedz na podstawie dokumentów:\n{context}\n\nPytanie: {question}\nOdpowiedź:"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()
