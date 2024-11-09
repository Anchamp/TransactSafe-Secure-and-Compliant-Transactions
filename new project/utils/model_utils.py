# utils/model_utils.py

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
import pandas as pd


class ModelEvaluator:
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate various model performance metrics"""
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_prob),
            'avg_precision': average_precision_score(y_true, y_prob),
        }

        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

        # Find optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_threshold_idx = np.argmax(f1_scores)
        metrics['optimal_threshold'] = thresholds[optimal_threshold_idx]

        return metrics

    def get_feature_importance(self, model, feature_names):
        """Get feature importance scores"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            return importance_df.sort_values('importance', ascending=False)
        return None


def save_model(model, filepath):
    """Save the trained model"""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load the trained model"""
    try:
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def generate_prediction_report(df, predictions, prediction_probs):
    """Generate a detailed prediction report"""
    df['predicted_label'] = predictions
    df['prediction_probability'] = prediction_probs[:, 1]  # Probability of fraudulent class

    # High-risk transactions (high probability of fraud)
    high_risk = df[df['prediction_probability'] > 0.8]

    report = {
        'total_transactions': len(df),
        'flagged_transactions': len(df[df['predicted_label'] == 1]),
        'high_risk_transactions': len(high_risk),
        'avg_transaction_risk': df['prediction_probability'].mean(),
        'high_risk_details': high_risk[['Timestamp', 'From Bank', 'To Bank',
                                        'Amount Paid', 'prediction_probability']]
    }

    return report