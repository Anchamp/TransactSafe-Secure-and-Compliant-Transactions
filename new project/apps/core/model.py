from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class Transaction:
    """Transaction data model"""
    row_key: str
    timestamp: datetime
    from_bank: str
    to_bank: str
    amount: float
    currency: str
    risk_score: float
    is_suspicious: bool

    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        return cls(
            row_key=data.get('row_key', ''),
            timestamp=datetime.strptime(data['Timestamp'], '%Y-%m-%d %H:%M:%S'),
            from_bank=data['From Bank'],
            to_bank=data['To Bank'],
            amount=float(data['Amount']),
            currency=data['Currency'],
            risk_score=float(data.get('risk_score', 0.0)),
            is_suspicious=bool(data.get('is_suspicious', False))
        )


@dataclass
class Alert:
    """Alert data model"""
    row_key: str
    transaction_key: str
    alert_date: datetime
    risk_score: float
    status: str = 'new'  # new, reviewed, closed


# apps/database/hbase_client.py
import happybase
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime
import uuid
import json
from ..core.models import Transaction, Alert


class HBaseClient:
    """HBase client for managing transaction and alert data"""

    def __init__(self, host: str = 'localhost', port: int = 9090):
        self.connection = happybase.Connection(host=host, port=port)
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure required tables exist"""
        tables = {b.decode('utf-8') for b in self.connection.tables()}

        if 'transactions' not in tables:
            self.connection.create_table(
                'transactions',
                {
                    'data': dict(),  # Basic transaction data
                    'risk': dict(),  # Risk-related information
                    'meta': dict()  # Metadata and audit info
                }
            )

        if 'alerts' not in tables:
            self.connection.create_table(
                'alerts',
                {
                    'data': dict(),  # Alert details
                    'transaction': dict(),  # Related transaction info
                    'meta': dict()  # Metadata and status
                }
            )

    def store_transactions(self, df: pd.DataFrame) -> List[str]:
        """Store transactions from DataFrame to HBase"""
        table = self.connection.table('transactions')
        batch = table.batch()
        row_keys = []

        for _, row in df.iterrows():
            row_key = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            row_keys.append(row_key)

            # Basic transaction data
            transaction_data = {
                'timestamp': str(row['Timestamp']),
                'from_bank': str(row['From Bank']),
                'to_bank': str(row['To Bank']),
                'amount': str(row['Amount']),
                'currency': str(row['Currency'])
            }

            # Risk information
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
        return row_keys

    def store_alert(self, transaction_key: str, risk_score: float) -> str:
        """Create and store alert for suspicious transaction"""
        table = self.connection.table('alerts')
        alert_key = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        alert_data = {
            b'data:alert_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S').encode(),
            b'data:risk_score': str(risk_score).encode(),
            b'transaction:transaction_key': transaction_key.encode(),
            b'meta:status': b'new'
        }

        table.put(alert_key.encode(), alert_data)
        return alert_key

    def get_recent_alerts(self, limit: int = 100) -> List[Alert]:
        """Retrieve recent alerts"""
        table = self.connection.table('alerts')
        alerts = []

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

    def get_transaction(self, row_key: str) -> Optional[Transaction]:
        """Retrieve single transaction by key"""
        table = self.connection.table('transactions')
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