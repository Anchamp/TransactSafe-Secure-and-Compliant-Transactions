# model_train.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import joblib
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModelAMLDetector:
    """Enhanced AML detection using multiple models and ensemble techniques."""

    def __init__(self,
                 contamination: float = 0.1,
                 random_state: int = 42,
                 n_estimators: int = 100):
        """Initialize detector with multiple models"""
        self.isolation_forest1 = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            max_samples='auto'
        )

        self.lof_model = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=20,
            novelty=True
        )

        self.dbscan_model = DBSCAN(
            eps=0.5,
            min_samples=5,
            n_jobs=-1
        )

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.fitted = False
        self.weights = {
            'iforest': 0.4,
            'lof': 0.35,
            'dbscan': 0.25
        }

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for AML detection"""
        features = pd.DataFrame()

        # Amount features - handle different possible column names
        amount_col = None
        for possible_col in ['Amount', 'grid_3x3Amount Received', 'Amount Received']:
            if possible_col in df.columns:
                amount_col = possible_col
                break

        if amount_col is None:
            raise ValueError("No amount column found in the dataset")

        # Process amount features
        features['amount'] = pd.to_numeric(df[amount_col], errors='coerce')
        features['amount_log'] = np.log1p(features['amount'])

        # Temporal features
        timestamps = pd.to_datetime(df['Timestamp'])
        features['hour'] = timestamps.dt.hour
        features['day_of_week'] = timestamps.dt.dayofweek
        features['is_weekend'] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
        features['is_business_hours'] = ((timestamps.dt.hour >= 9) &
                                         (timestamps.dt.hour <= 17)).astype(int)

        # Bank relationship features
        features['same_bank'] = (df['From Bank'] == df['To Bank']).astype(int)

        # Bank frequency features
        features['from_bank_freq'] = df['From Bank'].map(df['From Bank'].value_counts())
        features['to_bank_freq'] = df['To Bank'].map(df['To Bank'].value_counts())

        # Amount statistics by bank
        features['bank_avg_amount'] = df.groupby('From Bank')[amount_col].transform('mean')
        features['amount_to_avg_ratio'] = features['amount'] / features['bank_avg_amount'].fillna(1)

        # Encode categorical features
        categorical_cols = ['From Bank', 'To Bank', 'Currency']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                features = pd.concat([features, dummies], axis=1)

        # Handle missing values
        features = features.fillna(0)

        return features

    def fit(self, df: pd.DataFrame) -> 'MultiModelAMLDetector':
        """Fit all models in the ensemble"""
        try:
            logger.info("Starting model training...")

            # Engineer features
            features = self._engineer_features(df)

            # Scale features
            X_scaled = self.scaler.fit_transform(features)

            # Apply PCA
            X_pca = self.pca.fit_transform(X_scaled)
            logger.info(f"PCA explains {self.pca.explained_variance_ratio_.sum():.2%} of variance")

            # Fit models
            logger.info("Training Isolation Forest...")
            self.isolation_forest1.fit(X_pca)

            logger.info("Training LOF...")
            self.lof_model.fit(X_pca)

            logger.info("Training DBSCAN...")
            self.dbscan_model.fit(X_pca)

            self.fitted = True
            logger.info("Model training completed successfully")
            return self

        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate predictions using model ensemble"""
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")

        try:
            # Prepare features
            features = self._engineer_features(df)
            X_scaled = self.scaler.transform(features)
            X_pca = self.pca.transform(X_scaled)

            # Get individual model scores
            scores = {
                'iforest': -self.isolation_forest1.score_samples(X_pca),  # Negative for consistency
                'lof': -self.lof_model.score_samples(X_pca),
                'dbscan': self.dbscan_model.fit_predict(X_pca)
            }

            # Convert DBSCAN predictions to scores
            scores['dbscan'] = np.where(scores['dbscan'] == -1, 1, -1)

            # Calculate weighted ensemble score
            ensemble_scores = np.zeros(len(X_pca))
            for model, weight in self.weights.items():
                ensemble_scores += weight * scores[model]

            # Normalize scores to [0, 1] range
            ensemble_scores = (ensemble_scores - ensemble_scores.min()) / (
                        ensemble_scores.max() - ensemble_scores.min())

            return ensemble_scores, scores

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def generate_report(self, df: pd.DataFrame, predictions: np.ndarray, scores: Dict[str, np.ndarray]) -> Dict:
        """Generate detailed analysis report"""
        try:
            high_risk_mask = predictions > 0.7
            amount_col = [col for col in df.columns if 'amount' in col.lower() or 'Amount' in col][0]

            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_transactions': len(df),
                'high_risk_count': sum(high_risk_mask),
                'risk_stats': {
                    'mean': predictions.mean(),
                    'median': np.median(predictions),
                    'std': predictions.std()
                },
                'model_contributions': {
                    model: {
                        'mean_score': score.mean(),
                        'correlation': np.corrcoef(predictions, score)[0, 1]
                    }
                    for model, score in scores.items()
                },
                'high_risk_summary': {
                    'amount_stats': df.loc[high_risk_mask, amount_col].describe().to_dict(),
                    'top_banks': {
                        'from': df.loc[high_risk_mask, 'From Bank'].value_counts().head(5).to_dict(),
                        'to': df.loc[high_risk_mask, 'To Bank'].value_counts().head(5).to_dict()
                    }
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def save(self, filepath: str):
        """Save trained model"""
        if not self.fitted:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'MultiModelAMLDetector':
        """Load trained model"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    try:
        # Load training data
        DATA_PATH = "data/transactions.csv"
        MODEL_PATH = "models/aml_detector.joblib"

        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} transactions for training")

        # Print column names for debugging
        logger.info(f"Available columns: {df.columns.tolist()}")

        # Train model
        detector = MultiModelAMLDetector(contamination=0.1)
        detector.fit(df)

        # Generate predictions
        predictions, scores = detector.predict(df)

        # Generate report
        report = detector.generate_report(df, predictions, scores)

        # Print summary
        print("\nTraining Results:")
        print(f"Total Transactions: {report['total_transactions']}")
        print(f"High Risk Transactions: {report['high_risk_count']}")
        print(f"Average Risk Score: {report['risk_stats']['mean']:.3f}")

        # Save model
        detector.save(MODEL_PATH)

    except Exception as e:
        logger.error(f"Error in training script: {str(e)}")
        raise