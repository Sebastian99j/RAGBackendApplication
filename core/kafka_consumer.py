from confluent_kafka import Consumer
import json
from core.config import settings

def start_kafka_listener():
    consumer = Consumer({
        'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
        'group.id': 'llm-api-group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe([settings.KAFKA_TOPIC])

    print(f"[Kafka] Listening on topic '{settings.KAFKA_TOPIC}' at {settings.KAFKA_BOOTSTRAP_SERVERS}")

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print("Kafka error:", msg.error())
            continue

        data = json.loads(msg.value().decode('utf-8'))
        print(f"[Kafka] New message: {data}")
