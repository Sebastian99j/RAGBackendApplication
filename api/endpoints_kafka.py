from api.endpoints import router
from api.schemas import QueryRequest
from core.kafka_producer import send_kafka_message
import json
from services.openai_service import ask_openai


@router.post("/ask-openai-kafka")
def endpoint_openai(req: QueryRequest):
    answer = ask_openai(req.session_id, req.question, req.top_k, req.database)

    event = {
        "session_id": req.session_id,
        "question": req.question,
        "answer": answer,
        "model": "openai"
    }
    send_kafka_message(topic="llm-events", key=req.session_id, value=json.dumps(event))

    return {"model": "openai", "answer": answer}