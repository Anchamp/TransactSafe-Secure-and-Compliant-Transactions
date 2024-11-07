import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Tuple, Optional, Dict, List
from datetime import datetime
from feature_eng.preprocess import ColumnMapper, TransactionPreprocessor


# Rest of the code remains the same
class ColumnMapper:
    """Handles flexible column mapping and validation"""
    DEFAULT_MAPPINGS = {
        'timestamp': ['Timestamp'],
        'from_bank': ['From Bank'],
        'to_bank': ['To Bank'],
        'account': ['text_formatAccount'],
        'amount_received': ['Amount Received'],
        'amount_paid': ['Amount Paid'],
        'receiving_currency': ['text_formatReceiving Currency'],
        'payment_currency': ['text_formatPayment Currency'],
        'payment_format': ['text_formatPayment Format']
    }

    def __init__(self, custom_mappings: Optional[Dict[str, List[str]]] = None):
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        if custom_mappings:
            for key, values in custom_mappings.items():
                if key in self.mappings:
                    self.mappings[key].extend(values)
                else:
                    self.mappings[key] = values

    def find_column(self, df: pd.DataFrame, column_type: str) -> Optional[str]:
        """Find the actual column name in DataFrame for a given column type"""
        if column_type not in self.mappings:
            return None

        for possible_name in self.mappings[column_type]:
            if possible_name in df.columns:
                return possible_name
        return None

    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, str]]:
        """Validate DataFrame columns and return column mapping"""
        column_mapping = {}
        missing_columns = []

        print("Available columns in DataFrame:", df.columns.tolist())

        for col_type in self.mappings.keys():
            found_col = self.find_column(df, col_type)
            if found_col:
                column_mapping[col_type] = found_col
                print(f"Mapped {col_type} to {found_col}")
            else:
                missing_columns.append(col_type)
                print(f"Could not find mapping for {col_type}")

        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}", column_mapping

        return True, "", column_mapping


class UnsupervisedTransactionPreprocessor:
    """Preprocesses transaction data for unsupervised fraud detection"""

    def __init__(self, batch_size: int = 1000):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.batch_size = batch_size
        self.fitted = False
        self.column_mapper = ColumnMapper()
        self.column_mapping = None
        self.feature_names = None

    def _process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from timestamp"""
        timestamp_col = self.column_mapping['timestamp']
        try:
            dt = pd.to_datetime(df[timestamp_col])
            return pd.DataFrame({
                'hour': dt.dt.hour,
                'day': dt.dt.day,
                'month': dt.dt.month,
                'day_of_week': dt.dt.dayofweek,
                'is_weekend': dt.dt.dayofweek.isin([5, 6]).astype(int)
            })
        except Exception as e:
            warnings.warn(f"Error processing temporal features: {str(e)}")
            return pd.DataFrame({
                'hour': 0, 'day': 1, 'month': 1,
                'day_of_week': 0, 'is_weekend': 0
            }, index=df.index)

    def _process_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and create amount-related features"""
        received_col = self.column_mapping['amount_received']
        paid_col = self.column_mapping['amount_paid']

        amount_features = pd.DataFrame()
        amount_features['amount_received'] = pd.to_numeric(df[received_col], errors='coerce').fillna(0)
        amount_features['amount_paid'] = pd.to_numeric(df[paid_col], errors='coerce').fillna(0)

        # Derived features
        amount_features['amount_difference'] = amount_features['amount_paid'] - amount_features['amount_received']
        amount_features['amount_ratio'] = np.where(
            amount_features['amount_received'] != 0,
            amount_features['amount_paid'] / amount_features['amount_received'],
            0
        )

        return amount_features

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform data"""
        # Validate and get column mapping
        is_valid, error_msg, column_mapping = self.column_mapper.validate_columns(df)
        if not is_valid:
            raise ValueError(f"Invalid DataFrame structure: {error_msg}")

        self.column_mapping = column_mapping

        # Process features
        temporal_features = self._process_temporal_features(df)
        amount_features = self._process_amounts(df)

        # Process categorical features
        categorical_features = []
        for feature_type in ['from_bank', 'to_bank', 'receiving_currency', 'payment_currency', 'payment_format']:
            col_name = self.column_mapping[feature_type]
            dummies = pd.get_dummies(df[col_name], prefix=feature_type)
            categorical_features.append(dummies)

        # Combine all features
        features_df = pd.concat([temporal_features, amount_features] + categorical_features, axis=1)
        features_df = features_df.fillna(0)

        # Save feature names for future transformations
        self.feature_names = features_df.columns.tolist()

        # Fit and apply scaling
        scaled_features = self.scaler.fit_transform(features_df)

        # Fit and apply PCA
        reduced_features = self.pca.fit_transform(scaled_features)

        self.fitted = True
        return reduced_features

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Validate columns
        is_valid, error_msg, _ = self.column_mapper.validate_columns(df)
        if not is_valid:
            raise ValueError(f"Invalid DataFrame structure: {error_msg}")

        # Process features
        temporal_features = self._process_temporal_features(df)
        amount_features = self._process_amounts(df)

        # Process categorical features
        categorical_features = []
        for feature_type in ['from_bank', 'to_bank', 'receiving_currency', 'payment_currency', 'payment_format']:
            col_name = self.column_mapping[feature_type]
            dummies = pd.get_dummies(df[col_name], prefix=feature_type)
            categorical_features.append(dummies)

        # Combine all features
        features_df = pd.concat([temporal_features, amount_features] + categorical_features, axis=1)

        # Align features with training columns
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[self.feature_names]

        # Apply scaling and PCA
        scaled_features = self.scaler.transform(features_df)
        reduced_features = self.pca.transform(scaled_features)

        return reduced_features


class UnsupervisedFraudDetector:
    """Main class for unsupervised fraud detection"""

    def __init__(self, contamination: float = 0.1, random_state: int = 42, n_estimators: int = 100):
        self.preprocessor = UnsupervisedTransactionPreprocessor()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            n_jobs=-1
        )
        self.fitted = False

    def train(self, data_path: str, chunk_size: int = 1000) -> 'UnsupervisedFraudDetector':
        """Train the model using chunked data processing"""
        print("Starting unsupervised model training...")

        try:
            # Process first chunk to initialize preprocessor
            first_chunk = pd.read_csv(data_path, nrows=chunk_size)
            print("Sample columns in data:", first_chunk.columns.tolist())

            X = self.preprocessor.fit_transform(first_chunk)
            chunk_features = [X]

            # Process remaining data in chunks
            for chunk in pd.read_csv(data_path, skiprows=chunk_size, chunksize=chunk_size):
                X_chunk = self.preprocessor.transform(chunk)
                chunk_features.append(X_chunk)

            X_combined = np.vstack(chunk_features)

            # Train Isolation Forest
            print("Training Isolation Forest...")
            self.model.fit(X_combined)
            self.fitted = True

            # Calculate training statistics
            scores = self.model.score_samples(X_combined)
            predictions = self.model.predict(X_combined)
            anomaly_ratio = (predictions == -1).sum() / len(predictions)

            print(f"\nTraining completed:")
            print(f"Total samples processed: {len(X_combined)}")
            print(f"Detected anomaly ratio: {anomaly_ratio:.2%}")
            print(f"Score threshold: {np.percentile(scores, 10):.3f}")

            return self

        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise

    def predict(self, transaction_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomaly scores for transactions"""
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")

        X = self.preprocessor.transform(transaction_df)
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)

        return predictions, scores

    def save(self, model_path: str):
        """Save the trained model and preprocessor"""
        if not self.fitted:
            raise ValueError("Model must be trained before saving")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump((self.model, self.preprocessor), model_path)
        print(f"Model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str) -> 'UnsupervisedFraudDetector':
        """Load a trained model"""
        detector = cls()
        detector.model, detector.preprocessor = joblib.load(model_path)
        detector.fitted = True
        return detector


def generate_sample_data(output_path: str, n_samples: int = 10000):
    """Generate sample transaction data with correct column names"""
    np.random.seed(42)

    data = {
        'Timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
        'From Bank': np.random.choice(['Bank A', 'Bank B', 'Bank C', 'Bank D'], n_samples),
        'To Bank': np.random.choice(['Bank A', 'Bank B', 'Bank C', 'Bank D'], n_samples),
        'text_formatAccount': [f'ACC{str(i).zfill(6)}' for i in range(n_samples)],
        'Amount Received': np.random.lognormal(mean=7, sigma=1, size=n_samples),
        'Amount Paid': np.random.lognormal(mean=7, sigma=1, size=n_samples),
        'text_formatReceiving Currency': np.random.choice(['USD', 'EUR', 'GBP'], n_samples),
        'text_formatPayment Currency': np.random.choice(['USD', 'EUR', 'GBP'], n_samples),
        'text_formatPayment Format': np.random.choice(['SWIFT', 'WIRE', 'ACH'], n_samples)
    }

    # Add anomalous patterns
    anomaly_idx = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    data['Amount Received'][anomaly_idx] *= np.random.uniform(5, 10, size=len(anomaly_idx))
    data['Amount Paid'][anomaly_idx] *= np.random.uniform(5, 10, size=len(anomaly_idx))

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} sample transactions and saved to {output_path}")
    print("Generated columns:", df.columns.tolist())


if __name__ == '__main__':
    # Example usage
    DATA_PATH = "data/transactions.csv"
    MODEL_PATH = "models/fraud_detector.joblib"

    # Generate sample data if needed
    if not os.path.exists(DATA_PATH):
        generate_sample_data(DATA_PATH)
        print("\nSample data generated with correct column names")

    # Train model
    detector = UnsupervisedFraudDetector(contamination=0.1)
    detector.train(DATA_PATH)

    # Save the model
    detector.save(MODEL_PATH)

    # Example prediction
    test_data = pd.read_csv(DATA_PATH, nrows=5)
    predictions, scores = detector.predict(test_data)

    print("\nExample predictions:")
    print(pd.DataFrame({
        'Prediction': ['Anomaly' if p == -1 else 'Normal' for p in predictions],
        'Anomaly Score': scores.round(3)
    }))