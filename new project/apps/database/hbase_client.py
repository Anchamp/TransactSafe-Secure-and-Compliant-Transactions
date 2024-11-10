# apps/database/hbase_client.py
import happybase
import logging
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime
import uuid
from contextlib import contextmanager
from queue import Queue, Empty
import threading
from apps.core.model import Transaction, Alert

logger = logging.getLogger(__name__)


class HBaseConnectionPool:
    """Thread-safe HBase connection pool"""

    def __init__(self, host: str = 'localhost', port: int = 9090, size: int = 5):
        self.host = host
        self.port = port
        self.size = size
        self.pool = Queue(maxsize=size)
        self._lock = threading.Lock()
        self._initialize_pool()

    def _initialize_pool(self):
        """Initialize the connection pool"""
        for _ in range(self.size):
            try:
                connection = happybase.Connection(host=self.host, port=self.port)
                self.pool.put(connection)
            except Exception as e:
                logger.error(f"Error creating HBase connection: {e}")
                raise

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        connection = None
        try:
            connection = self.pool.get(timeout=5)
            yield connection
        except Empty:
            logger.error("Timeout waiting for available connection")
            raise Exception("No available connections")
        except Exception as e:
            logger.error(f"Error with HBase connection: {e}")
            raise
        finally:
            if connection:
                try:
                    # Test connection before returning to pool
                    connection.tables()
                    self.pool.put(connection)
                except Exception:
                    # If connection is broken, create new one
                    try:
                        connection = happybase.Connection(host=self.host, port=self.port)
                        self.pool.put(connection)
                    except Exception as e:
                        logger.error(f"Error recreating connection: {e}")


class HBaseClient:
    """Thread-safe HBase client with connection pooling"""

    def __init__(self, host: str = 'localhost', port: int = 9090, pool_size: int = 5):
        self.connection_pool = HBaseConnectionPool(host=host, port=port, size=pool_size)
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure required tables exist"""
        with self.connection_pool.get_connection() as connection:
            tables = {b.decode('utf-8') for b in connection.tables()}

            if 'transactions' not in tables:
                connection.create_table(
                    'transactions',
                    {
                        'data': dict(),
                        'risk': dict(),
                        'meta': dict()
                    }
                )
                logger.info("Created transactions table")

            if 'alerts' not in tables:
                connection.create_table(
                    'alerts',
                    {
                        'data': dict(),
                        'transaction': dict(),
                        'meta': dict()
                    }
                )
                logger.info("Created alerts table")

    def store_transactions(self, df: pd.DataFrame) -> List[str]:
        """Store transactions from DataFrame to HBase"""
        row_keys = []
        batch_size = 1000  # Process in batches

        with self.connection_pool.get_connection() as connection:
            table = connection.table('transactions')

            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i + batch_size]
                batch = table.batch(transaction=True)  # Use transaction for atomicity

                try:
                    for _, row in batch_df.iterrows():
                        row_key = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                        row_keys.append(row_key)

                        # Prepare data
                        transaction_data = {
                            'timestamp': str(row['Timestamp']),
                            'from_bank': str(row['From Bank']),
                            'to_bank': str(row['To Bank']),
                            'amount': str(row['Amount']),
                            'currency': str(row['Currency'])
                        }

                        risk_data = {
                            'risk_score': str(row.get('risk_score', 0.0)),
                            'is_suspicious': str(row.get('is_suspicious', False))
                        }

                        # Store in HBase
                        batch.put(
                            row_key.encode(),
                            {
                                b'data:' + k.encode(): v.encode()
                                for k, v in transaction_data.items()
                            }
                        )
                        batch.put(
                            row_key.encode(),
                            {
                                b'risk:' + k.encode(): v.encode()
                                for k, v in risk_data.items()
                            }
                        )

                    batch.send()
                    logger.info(f"Stored batch of {len(batch_df)} transactions")

                except Exception as e:
                    logger.error(f"Error storing transactions batch: {e}")
                    raise

        return row_keys

    def store_alert(self, transaction_key: str, risk_score: float) -> str:
        """Create and store alert for suspicious transaction"""
        with self.connection_pool.get_connection() as connection:
            table = connection.table('alerts')
            alert_key = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            try:
                alert_data = {
                    b'data:alert_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode(),
                    b'data:risk_score': str(risk_score).encode(),
                    b'transaction:transaction_key': transaction_key.encode(),
                    b'meta:status': b'new'
                }

                table.put(alert_key.encode(), alert_data)
                logger.info(f"Created alert: {alert_key}")
                return alert_key

            except Exception as e:
                logger.error(f"Error creating alert: {e}")
                raise

    def get_recent_alerts(self, limit: int = 100) -> List[Alert]:
        """Retrieve recent alerts"""
        with self.connection_pool.get_connection() as connection:
            table = connection.table('alerts')
            alerts = []

            try:
                for key, data in table.scan(limit=limit):
                    alert = Alert(
                        row_key=key.decode(),
                        transaction_key=data[b'transaction:transaction_key'].decode(),
                        alert_date=datetime.strptime(
                            data[b'data:alert_date'].decode(),
                            '%Y-%m-%d %H:%M:%S'
                        ),
                        risk_score=float(data[b'data:risk_score'].decode()),
                        status=data[b'meta:status'].decode()
                    )
                    alerts.append(alert)

                return sorted(alerts, key=lambda x: x.alert_date, reverse=True)

            except Exception as e:
                logger.error(f"Error retrieving alerts: {e}")
                raise

    def get_transaction(self, row_key: str) -> Optional[Transaction]:
        """Retrieve single transaction by key"""
        with self.connection_pool.get_connection() as connection:
            table = connection.table('transactions')

            try:
                row = table.row(row_key.encode())

                if not row:
                    return None

                return Transaction(
                    row_key=row_key,
                    timestamp=datetime.strptime(
                        row[b'data:timestamp'].decode(),
                        '%Y-%m-%d %H:%M:%S'
                    ),
                    from_bank=row[b'data:from_bank'].decode(),
                    to_bank=row[b'data:to_bank'].decode(),
                    amount=float(row[b'data:amount'].decode()),
                    currency=row[b'data:currency'].decode(),
                    risk_score=float(row[b'risk:risk_score'].decode()),
                    is_suspicious=row[b'risk:is_suspicious'].decode() == 'True'
                )

            except Exception as e:
                logger.error(f"Error retrieving transaction: {e}")
                raise

    def test_connection(self) -> bool:
        """Test if HBase connection is working"""
        try:
            with self.connection_pool.get_connection() as connection:
                connection.tables()
                return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Usage example:
if __name__ == "__main__":
    # Test connection
    client = HBaseClient(host='localhost', port=9090)
    if client.test_connection():
        print("HBase connection successful!")
    else:
        print("HBase connection failed!")