import json
from kafka import KafkaProducer
from kafka.errors import KafkaError
import time
import random

# Initialize Kafka producer
def create_producer(bootstrap_servers=['localhost:9092']):
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def produce_transaction(producer, topic="transactions"):
    # Sample transaction data
    transaction = {
        "transaction_id": random.randint(1000, 9999),
        "timestamp": time.time(),
        "from_bank": "Bank A",
        "to_bank": "Bank B",
        "amount": round(random.uniform(100, 5000), 2)
    }
    try:
        producer.send(topic, transaction)
        producer.flush()
        print(f"Transaction sent: {transaction}")
    except KafkaError as e:
        print(f"Failed to send transaction: {e}")

if __name__ == "__main__":
    producer = create_producer()
    while True:
        produce_transaction(producer)
        time.sleep(1)  # Sending one transaction per second
