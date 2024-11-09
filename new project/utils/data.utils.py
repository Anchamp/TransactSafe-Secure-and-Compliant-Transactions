# utils/data_utils.py
import pandas as pd
import numpy as np
from datetime import datetime


def validate_transaction_data(transaction):
    """Validate transaction data format"""
    required_fields = [
        'Timestamp',
        'grid_3x3From Bank',
        'text_formatAccount',
        'grid_3x3To Bank',
        'grid_3x3Amount Received',
        'text_formatReceiving Currency',
        'grid_3x3Amount Paid',
        'text_formatPayment Currency',
        'text_formatPayment Format'
    ]

    missing_fields = [field for field in required_fields if field not in transaction]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Validate amount fields
    try:
        float(transaction['grid_3x3Amount Received'])
        float(transaction['grid_3x3Amount Paid'])
    except ValueError:
        raise ValueError("Invalid amount format")

    # Validate timestamp
    try:
        datetime.strptime(transaction['Timestamp'], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        raise ValueError("Invalid timestamp format")

    return True


def calculate_transaction_metrics(df):
    """Calculate key metrics from transaction data"""
    metrics = {
        'total_transactions': len(df),
        'total_amount_received': df['grid_3x3Amount Received'].sum(),
        'total_amount_paid': df['grid_3x3Amount Paid'].sum(),
        'unique_banks': {
            'from': df['grid_3x3From Bank'].nunique(),
            'to': df['grid_3x3To Bank'].nunique()
        },
        'currency_distribution': {
            'receiving': df['text_formatReceiving Currency'].value_counts().to_dict(),
            'payment': df['text_formatPayment Currency'].value_counts().to_dict()
        },
        'payment_format_distribution': df['text_formatPayment Format'].value_counts().to_dict()
    }
    return metrics


def generate_alert_report(df, risk_scores):
    """Generate detailed alert report"""
    df['risk_score'] = risk_scores
    high_risk = df[df['risk_score'] > 0.7]

    report = {
        'total_transactions': len(df),
        'high_risk_transactions': len(high_risk),
        'average_risk_score': df['risk_score'].mean(),
        'risk_by_bank': df.groupby('grid_3x3From Bank')['risk_score'].mean().to_dict(),
        'high_risk_details': high_risk[[
            'Timestamp',
            'grid_3x3From Bank',
            'grid_3x3To Bank',
            'grid_3x3Amount Paid',
            'text_formatPayment Currency',
            'risk_score'
        ]].to_dict('records')
    }
    return report