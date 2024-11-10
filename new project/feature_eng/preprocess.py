# feature_eng/preprocess.py
'''import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Iterator, Dict, List, Optional, Tuple
import gc

class ColumnMapper:
    """Handles flexible column mapping and validation"""
    DEFAULT_MAPPINGS = {
        'Timestamp': ['Timestamp', 'timestamp'],
        'From Bank': ['From Bank', 'grid_3x3From Bank', 'from_bank'],
        'To Bank': ['To Bank', 'grid_3x3To Bank', 'to_bank'],
        'text_formatAccount': ['text_formatAccount', 'account'],
        'Amount Received': ['Amount Received', 'grid_3x3Amount Received', 'amount_received'],
        'Amount Paid': ['Amount Paid', 'grid_3x3Amount Paid', 'amount_paid'],
        'text_formatReceiving Currency': ['text_formatReceiving Currency', 'receiving_currency'],
        'text_formatPayment Currency': ['text_formatPayment Currency', 'payment_currency'],
        'text_formatPayment Format': ['text_formatPayment Format', 'payment_format']
    }

    def __init__(self, custom_mappings: Optional[Dict[str, List[str]]] = None):
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        if custom_mappings:
            for key, values in custom_mappings.items():
                if key in self.mappings:
                    self.mappings[key].extend(values)
                else:
                    self.mappings[key] = values

    def find_column(self, df: pd.DataFrame, standard_name: str) -> Optional[str]:
        """Find the actual column name in DataFrame for a given standard name"""
        if standard_name not in self.mappings:
            return None

        for possible_name in self.mappings[standard_name]:
            if possible_name in df.columns:
                return possible_name
        return None

    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, str]]:
        """Validate DataFrame columns and return column mapping"""
        column_mapping = {}
        missing_columns = []

        for standard_name in self.mappings.keys():
            found_col = self.find_column(df, standard_name)
            if found_col:
                column_mapping[standard_name] = found_col
            else:
                missing_columns.append(standard_name)

        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}", column_mapping

        return True, "", column_mapping

class TransactionPreprocessor:
    def __init__(self, batch_size: int = 1000):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.fitted = False
        self.column_mapper = ColumnMapper()
        self.column_mapping = None
'''
# feature_eng/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColumnMapper:
    """
    Flexible column mapping system for transaction data processing.
    Handles various column name formats and validates DataFrame structure.
    """
    DEFAULT_MAPPINGS = {
        'Timestamp': ['Timestamp', 'timestamp', 'date', 'transaction_date', 'Date'],
        'From Bank': ['From Bank', 'from_bank', 'grid_3x3From Bank', 'sender_bank', 'source_bank'],
        'To Bank': ['To Bank', 'to_bank', 'grid_3x3To Bank', 'receiver_bank', 'destination_bank'],
        'Account': ['Account', 'text_formatAccount', 'account_number', 'account_id'],
        'Amount Received': ['Amount Received', 'amount_received', 'grid_3x3Amount Received', 'received_amount'],
        'Amount Paid': ['Amount Paid', 'amount_paid', 'grid_3x3Amount Paid', 'paid_amount'],
        'Currency': ['Currency', 'currency', 'transaction_currency'],
        'Payment Format': ['Payment Format', 'text_formatPayment Format', 'payment_method', 'transaction_type']
    }

    def __init__(self, custom_mappings: Optional[Dict[str, List[str]]] = None):
        """Initialize with optional custom column mappings"""
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        if custom_mappings:
            for key, values in custom_mappings.items():
                if key in self.mappings:
                    self.mappings[key].extend(values)
                else:
                    self.mappings[key] = values

    def find_column(self, df: pd.DataFrame, standard_name: str) -> Optional[str]:
        """Find matching column name in DataFrame"""
        if standard_name not in self.mappings:
            return None

        for possible_name in self.mappings[standard_name]:
            if possible_name in df.columns:
                return possible_name
        return None

    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, str]]:
        """Validate DataFrame columns and return mapping"""
        column_mapping = {}
        missing_columns = []

        for standard_name in self.mappings.keys():
            found_col = self.find_column(df, standard_name)
            if found_col:
                column_mapping[standard_name] = found_col
            else:
                missing_columns.append(standard_name)

        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}", column_mapping

        return True, "", column_mapping


class TransactionPreprocessor:
    """
    Advanced transaction data preprocessor with comprehensive feature engineering.
    Handles data cleaning, feature extraction, and transformation for AML detection.
    """

    def __init__(self,
                 custom_mappings: Optional[Dict[str, List[str]]] = None,
                 pca_components: float = 0.95,
                 random_state: int = 42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=random_state)
        self.column_mapper = ColumnMapper(custom_mappings)
        self.column_mapping = None
        self.feature_names = None
        self.fitted = False
        self.random_state = random_state

    def _process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and engineer temporal features"""
        try:
            timestamps = pd.to_datetime(df[self.column_mapping['Timestamp']])

            features = pd.DataFrame({
                'hour': timestamps.dt.hour,
                'day': timestamps.dt.day,
                'month': timestamps.dt.month,
                'day_of_week': timestamps.dt.dayofweek,
                'is_weekend': timestamps.dt.dayofweek.isin([5, 6]).astype(int),
                'is_business_hours': ((timestamps.dt.hour >= 9) &
                                      (timestamps.dt.hour <= 17)).astype(int),
                'is_night': ((timestamps.dt.hour >= 22) |
                             (timestamps.dt.hour <= 5)).astype(int),
                'week_of_year': timestamps.dt.isocalendar().week,
                'day_of_month': timestamps.dt.day,
                'quarter': timestamps.dt.quarter
            })

            return features

        except Exception as e:
            logger.warning(f"Error processing temporal features: {str(e)}")
            return pd.DataFrame(index=df.index).assign(
                hour=0, day=1, month=1, day_of_week=0,
                is_weekend=0, is_business_hours=0, is_night=0,
                week_of_year=1, day_of_month=1, quarter=1
            )

    def _process_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and engineer amount-related features"""
        features = pd.DataFrame()

        # Basic amount features
        amount_received = pd.to_numeric(df[self.column_mapping['Amount Received']], errors='coerce').fillna(0)
        amount_paid = pd.to_numeric(df[self.column_mapping['Amount Paid']], errors='coerce').fillna(0)

        features['amount_received'] = amount_received
        features['amount_paid'] = amount_paid

        # Derived features
        features['amount_difference'] = amount_paid - amount_received
        features['amount_ratio'] = np.where(amount_received != 0,
                                            amount_paid / amount_received,
                                            0)

        # Statistical features
        features['amount_log_received'] = np.log1p(amount_received)
        features['amount_log_paid'] = np.log1p(amount_paid)
        features['amount_squared'] = amount_paid ** 2
        features['amount_cube_root'] = np.cbrt(amount_paid)

        # Transaction size indicators
        mean_amount = amount_paid.mean()
        std_amount = amount_paid.std()
        features['is_large_transaction'] = (amount_paid > (mean_amount + 2 * std_amount)).astype(int)
        features['is_small_transaction'] = (amount_paid < (mean_amount - std_amount)).astype(int)

        return features

    def _process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process categorical features with enhanced encoding"""
        categorical_features = [
            'From Bank', 'To Bank', 'Currency', 'Payment Format'
        ]

        dummies_list = []

        for feature in categorical_features:
            if feature in self.column_mapping:
                col = self.column_mapping[feature]

                # Basic one-hot encoding
                dummies = pd.get_dummies(df[col], prefix=feature)
                dummies_list.append(dummies)

                # Additional derived features
                if feature in ['From Bank', 'To Bank']:
                    # Bank relationship features
                    df[f'{feature}_freq'] = df[col].map(df[col].value_counts())
                    dummies_list.append(pd.DataFrame({
                        f'{feature}_freq': df[f'{feature}_freq']
                    }))

        # Combine all categorical features
        return pd.concat(dummies_list, axis=1)

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different feature types"""
        features = pd.DataFrame()

        # Bank-Amount interactions
        from_bank_freq = df[self.column_mapping['From Bank']].map(
            df[self.column_mapping['From Bank']].value_counts()
        )
        amount_paid = pd.to_numeric(df[self.column_mapping['Amount Paid']], errors='coerce')

        features['bank_amount_interaction'] = from_bank_freq * amount_paid
        features['bank_amount_ratio'] = amount_paid / (from_bank_freq + 1)

        # Time-Amount interactions
        hour = pd.to_datetime(df[self.column_mapping['Timestamp']]).dt.hour
        features['hour_amount_interaction'] = hour * amount_paid

        return features

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit preprocessor and transform data"""
        try:
            logger.info("Starting feature engineering...")

            # Validate columns
            is_valid, error_msg, column_mapping = self.column_mapper.validate_columns(df)
            if not is_valid:
                raise ValueError(f"Invalid DataFrame structure: {error_msg}")

            self.column_mapping = column_mapping

            # Generate features
            temporal_features = self._process_temporal_features(df)
            amount_features = self._process_amount_features(df)
            categorical_features = self._process_categorical_features(df)
            interaction_features = self._create_interaction_features(df)

            # Combine all features
            features_df = pd.concat([
                temporal_features,
                amount_features,
                categorical_features,
                interaction_features
            ], axis=1).fillna(0)

            # Store feature names
            self.feature_names = features_df.columns.tolist()

            # Scale features
            X_scaled = self.scaler.fit_transform(features_df)

            # Apply PCA
            X_reduced = self.pca.fit_transform(X_scaled)
            logger.info(f"Retained {self.pca.n_components_} components explaining "
                        f"{self.pca.explained_variance_ratio_.sum():.2%} variance")

            self.fitted = True
            return X_reduced

        except Exception as e:
            logger.error(f"Error in fit_transform: {str(e)}")
            raise

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        try:
            # Validate columns
            is_valid, error_msg, _ = self.column_mapper.validate_columns(df)
            if not is_valid:
                raise ValueError(f"Invalid DataFrame structure: {error_msg}")

            # Generate features
            temporal_features = self._process_temporal_features(df)
            amount_features = self._process_amount_features(df)
            categorical_features = self._process_categorical_features(df)
            interaction_features = self._create_interaction_features(df)

            # Combine features
            features_df = pd.concat([
                temporal_features,
                amount_features,
                categorical_features,
                interaction_features
            ], axis=1).fillna(0)

            # Align features with training columns
            for col in self.feature_names:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[self.feature_names]

            # Apply scaling and PCA
            X_scaled = self.scaler.transform(features_df)
            X_reduced = self.pca.transform(X_scaled)

            return X_reduced

        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance based on PCA components"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before getting feature importance")

        try:
            # Calculate absolute importance of each feature
            feature_importance = pd.DataFrame(
                np.abs(self.pca.components_),
                columns=self.feature_names
            )

            # Weight by explained variance ratio
            weighted_importance = feature_importance.mul(
                self.pca.explained_variance_ratio_[:, np.newaxis]
            )

            # Get mean importance across components
            mean_importance = weighted_importance.mean()

            return pd.DataFrame({
                'feature': mean_importance.index,
                'importance': mean_importance.values
            }).sort_values('importance', ascending=False)

        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise


if __name__ == "__main__":
    # Example usage
    try:
        # Load sample data
        df = pd.read_csv("data/transactions.csv")

        # Initialize preprocessor
        preprocessor = TransactionPreprocessor()

        # Fit and transform data
        X = preprocessor.fit_transform(df)

        # Get feature importance
        importance_df = preprocessor.get_feature_importance()
        print("\nTop 10 most important features:")
        print(importance_df.head(10))

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise