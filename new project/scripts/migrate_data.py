# scripts/migrate_data.py
import pandas as pd
import logging
from apps.database.hbase_client import HBaseClient
from datetime import datetime
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataMigrator:
    def __init__(self, hbase_host='localhost', hbase_port=9090):
        self.client = HBaseClient(host=hbase_host, port=hbase_port)

    def validate_transaction_data(self, df: pd.DataFrame) -> bool:
        """Validate transaction DataFrame structure"""
        required_columns = [
            'Timestamp',
            'From Bank',
            'To Bank',
            'Amount',
            'Currency'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        return True

    def validate_alert_data(self, df: pd.DataFrame) -> bool:
        """Validate alert DataFrame structure"""
        required_columns = [
            'alert_date',
            'transaction_id',
            'risk_score'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        return True

    def migrate_transactions(self, csv_path: str, batch_size: int = 1000) -> bool:
        """Migrate transactions from CSV to HBase"""
        try:
            logger.info(f"Starting migration of transactions from {csv_path}")

            # Read CSV in chunks
            chunk_iterator = pd.read_csv(
                csv_path,
                chunksize=batch_size,
                parse_dates=['Timestamp']
            )

            total_records = 0
            for chunk_idx, chunk in enumerate(chunk_iterator, 1):
                if not self.validate_transaction_data(chunk):
                    logger.error(f"Invalid data in chunk {chunk_idx}")
                    continue

                # Store chunk in HBase
                row_keys = self.client.store_transactions(chunk)
                total_records += len(row_keys)
                logger.info(f"Processed chunk {chunk_idx}: {len(row_keys)} records")

            logger.info(f"Migration completed: {total_records} total records")
            return True

        except Exception as e:
            logger.error(f"Error during transaction migration: {str(e)}")
            return False

    def migrate_alerts(self, csv_path: str, batch_size: int = 1000) -> bool:
        """Migrate alerts from CSV to HBase"""
        try:
            logger.info(f"Starting migration of alerts from {csv_path}")

            # Read CSV in chunks
            chunk_iterator = pd.read_csv(
                csv_path,
                chunksize=batch_size,
                parse_dates=['alert_date']
            )

            total_records = 0
            for chunk_idx, chunk in enumerate(chunk_iterator, 1):
                if not self.validate_alert_data(chunk):
                    logger.error(f"Invalid data in chunk {chunk_idx}")
                    continue

                # Process each alert in chunk
                for _, row in chunk.iterrows():
                    alert_key = self.client.store_alert(
                        str(row['transaction_id']),
                        float(row['risk_score'])
                    )
                    total_records += 1

                logger.info(f"Processed chunk {chunk_idx}")

            logger.info(f"Migration completed: {total_records} total records")
            return True

        except Exception as e:
            logger.error(f"Error during alert migration: {str(e)}")
            return False


def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)

    # Initialize migrator
    migrator = DataMigrator()

    # Migrate transactions
    if os.path.exists('data/fraud_transactions.csv'):
        success = migrator.migrate_transactions('data/fraud_transactions.csv')
        if success:
            logger.info("Transaction migration successful")
        else:
            logger.error("Transaction migration failed")

    # Migrate alerts
    if os.path.exists('data/alerts.csv'):
        success = migrator.migrate_alerts('data/alerts.csv')
        if success:
            logger.info("Alert migration successful")
        else:
            logger.error("Alert migration failed")


if __name__ == "__main__":
    main()
