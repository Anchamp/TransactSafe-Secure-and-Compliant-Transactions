import json
import logging
from kafka import KafkaConsumer
from ..feature_engineering.preprocess import preprocess_transaction
from ..utils.model_utils import load_model

def consume_transactions(consumer, model):
    for message in consumer:
        transaction = json.loads(message.value)
        X = preprocess_transaction(transaction)
        prediction = model.predict([X])[0]
        logging.info(f"Transaction at {transaction['timestamp']} from bank {transaction['from_bank']} to bank {transaction['to_bank']} predicted as {'fraudulent' if prediction else 'legitimate'}")