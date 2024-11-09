import json
from kafka import KafkaProducer
from kafka.errors import KafkaError

def create_producer(bootstrap_servers=['localhost:9092']):
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        retries=5  # Retry sending messages up to 5 times
    )

def produce_transaction(producer, topic, transaction):
    try:
        future = producer.send(topic, transaction)
        future.add_callback(on_send_success)
        future.add_errback(on_send_error)
    except KafkaError as e:
        print(f"Failed to send transaction: {e}")
    finally:
        producer.flush()

def on_send_success(record_metadata):
    print(f"Message sent to {record_metadata.topic} partition {record_metadata.partition} at offset {record_metadata.offset}")

def on_send_error(excp):
    print(f"Error occurred: {excp}")
