import redis
import json
import os

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)


def get_chat_history(session_id: str):
    history = r.get(session_id)
    return json.loads(history) if history else []


def add_message_to_history(session_id: str, user_message: str, assistant_reply: str):
    history = get_chat_history(session_id)
    history.append({"user": user_message, "assistant": assistant_reply})
    r.set(session_id, json.dumps(history))


def reset_chat_history(session_id: str):
    r.delete(session_id)


def build_chat_prompt(history, question: str):
    dialogue = "\n".join(
        f"Użytkownik: {h['user']}\nAsystent: {h['assistant']}" for h in history
    )
    return f"{dialogue}\nUżytkownik: {question}\nAsystent:"
