from confluent_kafka import Producer
from core.config import settings

producer = Producer({'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS})

def send_kafka_message(topic: str, key: str, value: str):
    producer.produce(topic=topic, key=key, value=value)
    producer.flush()
