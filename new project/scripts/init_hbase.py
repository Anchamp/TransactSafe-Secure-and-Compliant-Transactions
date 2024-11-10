# scripts/init_hbase.py
import happybase
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@contextmanager
def get_connection(host='localhost', port=9090):
    """Context manager for HBase connection"""
    connection = None
    try:
        connection = happybase.Connection(host=host, port=port)
        yield connection
    except Exception as e:
        logger.error(f"Error connecting to HBase: {e}")
        raise
    finally:
        if connection:
            try:
                connection.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")


def init_hbase():
    """Initialize HBase with required tables"""
    try:
        with get_connection() as connection:
            # Get existing tables
            tables = {b.decode('utf-8') for b in connection.tables()}

            # Create transactions table if not exists
            if 'transactions' not in tables:
                connection.create_table(
                    'transactions',
                    {
                        'data': dict(),  # Basic transaction data
                        'risk': dict(),  # Risk-related information
                        'meta': dict()  # Metadata and audit info
                    }
                )
                logger.info("Created transactions table")

            # Create alerts table if not exists
            if 'alerts' not in tables:
                connection.create_table(
                    'alerts',
                    {
                        'data': dict(),  # Alert details
                        'transaction': dict(),  # Related transaction info
                        'meta': dict()  # Metadata and status
                    }
                )
                logger.info("Created alerts table")

            logger.info("HBase initialization completed successfully")

    except Exception as e:
        logger.error(f"Error initializing HBase: {e}")
        raise


if __name__ == "__main__":
    init_hbase()