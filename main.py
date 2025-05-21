from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.endpoints import router as api_router
import threading
from core.kafka_consumer import start_kafka_listener

app = FastAPI(title="Multi-LLM RAG API")

threading.Thread(target=start_kafka_listener, daemon=True).start()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/healthcheck")
def healthcheck():
    return {"status": "success"}
