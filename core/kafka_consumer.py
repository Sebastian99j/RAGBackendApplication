from confluent_kafka import Consumer
import json

def start_kafka_listener():
    consumer = Consumer({
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'llm-api-group',
        'auto.offset.reset': 'earliest'
    })
    consumer.subscribe(['llm-events'])

    while True:
        msg = consumer.poll(1.0)
        if msg is None: continue
        if msg.error():
            print("Kafka error:", msg.error())
            continue

        data = json.loads(msg.value().decode('utf-8'))
        print(f"[Kafka] New message: {data}")
