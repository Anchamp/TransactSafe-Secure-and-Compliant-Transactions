# models/enhanced_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import joblib
import os
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedUnsupervisedDetector:
    """
    Enhanced unsupervised fraud detection using multiple models
    and ensemble techniques
    """

    def __init__(self,
                 contamination: float = 0.1,
                 random_state: int = 42,
                 n_estimators: int = 100,
                 n_neighbors: int = 20):
        """
        Initialize the enhanced detector with multiple models

        Args:
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
            n_estimators: Number of trees in Isolation Forest
            n_neighbors: Number of neighbors for LOF
        """
        # Initialize multiple detection models
        self.isolation_forest1 = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples='auto'
        )

        self.isolation_forest2 = IsolationForest(
            contamination=contamination,
            random_state=random_state + 1,
            n_estimators=n_estimators,
            max_samples=256
        )

        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )

        self.dbscan = DBSCAN(
            eps=0.5,
            min_samples=5,
            n_jobs=-1
        )

        # Initialize preprocessing components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.fitted = False

    def preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data including feature engineering

        Args:
            df: Input DataFrame

        Returns:
            Processed features as numpy array
        """
        # Extract basic features
        features = pd.DataFrame()

        # Amount-based features
        features['amount_received'] = pd.to_numeric(df['Amount Received'], errors='coerce')
        features['amount_paid'] = pd.to_numeric(df['Amount Paid'], errors='coerce')
        features['amount_diff'] = features['amount_paid'] - features['amount_received']
        features['amount_ratio'] = features['amount_paid'] / features['amount_received'].replace(0, 1)

        # Time-based features
        timestamps = pd.to_datetime(df['Timestamp'])
        features['hour'] = timestamps.dt.hour
        features['day_of_week'] = timestamps.dt.dayofweek
        features['is_weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)

        # Bank relationship features
        features['same_bank'] = (df['From Bank'] == df['To Bank']).astype(int)

        # Currency features
        features['same_currency'] = (
                df['text_formatReceiving Currency'] == df['text_formatPayment Currency']
        ).astype(int)

        # Encode categorical variables
        categorical_features = [
            'From Bank', 'To Bank',
            'text_formatReceiving Currency',
            'text_formatPayment Currency',
            'text_formatPayment Format'
        ]

        for feature in categorical_features:
            dummies = pd.get_dummies(df[feature], prefix=feature)
            features = pd.concat([features, dummies], axis=1)

        # Fill missing values and scale
        features = features.fillna(0)
        if not self.fitted:
            features_scaled = self.scaler.fit_transform(features)
            self.feature_names = features.columns
        else:
            features_scaled = self.scaler.transform(features)

        return features_scaled

    def fit(self, df: pd.DataFrame) -> 'EnhancedUnsupervisedDetector':
        """
        Fit all models in the ensemble

        Args:
            df: Input DataFrame

        Returns:
            Self for method chaining
        """
        logger.info("Starting model training...")

        try:
            # Preprocess data
            X = self.preprocess_data(df)

            # Apply PCA if not fitted
            if not self.fitted:
                X_reduced = self.pca.fit_transform(X)
                logger.info(
                    f"Retained {self.pca.n_components_} components explaining {self.pca.explained_variance_ratio_.sum():.2%} variance")
            else:
                X_reduced = self.pca.transform(X)

            # Fit all models
            logger.info("Fitting Isolation Forest 1...")
            self.isolation_forest1.fit(X_reduced)

            logger.info("Fitting Isolation Forest 2...")
            self.isolation_forest2.fit(X_reduced)

            logger.info("Fitting LOF...")
            self.lof.fit(X_reduced)

            logger.info("Fitting DBSCAN...")
            self.dbscan.fit(X_reduced)

            self.fitted = True
            logger.info("Model training completed successfully")

            return self

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Generate predictions using the ensemble of models

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (ensemble predictions, individual model scores)
        """
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")

        # Preprocess data
        X = self.preprocess_data(df)
        X_reduced = self.pca.transform(X)

        # Get predictions from each model
        scores = {
            'iforest1': self.isolation_forest1.score_samples(X_reduced),
            'iforest2': self.isolation_forest2.score_samples(X_reduced),
            'lof': -self.lof.score_samples(X_reduced),  # Negative for consistency
            'dbscan': np.where(self.dbscan.fit_predict(X_reduced) == -1, -1, 1)
        }

        # Combine predictions (weighted average)
        weights = {
            'iforest1': 0.3,
            'iforest2': 0.3,
            'lof': 0.25,
            'dbscan': 0.15
        }

        ensemble_scores = sum(score * weights[model] for model, score in scores.items())

        # Convert to binary predictions (-1 for anomaly, 1 for normal)
        threshold = np.percentile(ensemble_scores, 90)  # Adjustable threshold
        predictions = np.where(ensemble_scores < threshold, -1, 1)

        return predictions, scores

    def save(self, filepath: str):
        """Save the trained model"""
        if not self.fitted:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'EnhancedUnsupervisedDetector':
        """Load a trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


def generate_evaluation_report(predictions: np.ndarray, scores: Dict[str, np.ndarray], df: pd.DataFrame) -> Dict:
    """
    Generate a detailed evaluation report

    Args:
        predictions: Binary predictions from the ensemble
        scores: Dictionary of scores from individual models
        df: Original DataFrame

    Returns:
        Dictionary containing evaluation metrics and summaries
    """
    n_samples = len(predictions)
    n_anomalies = (predictions == -1).sum()

    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_transactions': n_samples,
        'flagged_transactions': n_anomalies,
        'anomaly_rate': n_anomalies / n_samples,
        'model_correlations': {},
        'high_risk_examples': []
    }

    # Calculate correlations between model scores
    score_df = pd.DataFrame(scores)
    report['model_correlations'] = score_df.corr().to_dict()

    # Get high risk examples
    high_risk_idx = np.where(predictions == -1)[0]
    for idx in high_risk_idx[:10]:  # Top 10 examples
        risk_scores = {model: scores[model][idx] for model in scores.keys()}
        report['high_risk_examples'].append({
            'index': int(idx),
            'timestamp': df.iloc[idx]['Timestamp'],
            'amount_paid': float(df.iloc[idx]['Amount Paid']),
            'risk_scores': risk_scores
        })

    return report


if __name__ == "__main__":
    # Example usage
    DATA_PATH = "data/transactions.csv"
    MODEL_PATH = "models/enhanced_detector.joblib"

    # Load data
    df = pd.read_csv(DATA_PATH)

    # Create and train model
    detector = EnhancedUnsupervisedDetector(contamination=0.1)
    detector.fit(df)

    # Generate predictions
    predictions, scores = detector.predict(df)

    # Generate report
    report = generate_evaluation_report(predictions, scores, df)

    # Print summary
    print("\nDetection Summary:")
    print(f"Total Transactions: {report['total_transactions']}")
    print(f"Flagged as Anomalous: {report['flagged_transactions']} ({report['anomaly_rate']:.2%})")

    # Save model
    detector.save(MODEL_PATH)