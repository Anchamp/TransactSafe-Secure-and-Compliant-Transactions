import json
from kafka import KafkaProducer

def produce_transaction(producer, topic, transaction):
    producer.send(topic, json.dumps(transaction).encode('utf-8'))
    producer.flush()