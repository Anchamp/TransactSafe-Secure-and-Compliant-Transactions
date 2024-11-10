# scripts/verify_migration.py
import pandas as pd
import logging
from app.database.hbase_client import HBaseClient
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataVerifier:
    def __init__(self, hbase_host='localhost', hbase_port=9090):
        self.client = HBaseClient(host=hbase_host, port=hbase_port)

    def verify_transactions(self, csv_path: str) -> Tuple[bool, str]:
        """Verify transaction data migration"""
        try:
            # Read original CSV
            df_original = pd.read_csv(csv_path)
            original_count = len(df_original)

            # Get migrated data
            transactions = []
            with self.client.connection_pool.get_connection() as connection:
                table = connection.table('transactions')
                for key, row in table.scan():
                    transactions.append({
                        'timestamp': row[b'data:timestamp'].decode(),
                        'from_bank': row[b'data:from_bank'].decode(),
                        'to_bank': row[b'data:to_bank'].decode(),
                        'amount': float(row[b'data:amount'].decode()),
                        'currency': row[b'data:currency'].decode()
                    })

            migrated_count = len(transactions)

            # Compare counts
            if original_count != migrated_count:
                return False, f"Count mismatch: Original={original_count}, Migrated={migrated_count}"

            logger.info("Transaction migration verified successfully")
            return True, "Verification successful"

        except Exception as e:
            return False, f"Verification failed: {str(e)}"

    def verify_alerts(self, csv_path: str) -> Tuple[bool, str]:
        """Verify alert data migration"""
        try:
            # Read original CSV
            df_original = pd.read_csv(csv_path)
            original_count = len(df_original)

            # Get migrated alerts
            alerts = self.client.get_recent_alerts(limit=original_count + 100)
            migrated_count = len(alerts)

            # Compare counts
            if original_count != migrated_count:
                return False, f"Count mismatch: Original={original_count}, Migrated={migrated_count}"

            logger.info("Alert migration verified successfully")
            return True, "Verification successful"

        except Exception as e:
            return False, f"Verification failed: {str(e)}"


def main():
    verifier = DataVerifier()

    # Verify transactions
    logger.info("Verifying transactions...")
    success, message = verifier.verify_transactions('data/fraud_transactions.csv')
    logger.info(f"Transaction verification: {message}")

    # Verify alerts
    logger.info("Verifying alerts...")
    success, message = verifier.verify_alerts('data/alerts.csv')
    logger.info(f"Alert verification: {message}")


if __name__ == "__main__":
    main()