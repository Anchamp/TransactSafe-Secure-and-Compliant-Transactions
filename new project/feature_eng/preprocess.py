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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ColumnMapper:
    """
    Flexible column mapping system that handles various column name formats
    and validates DataFrame structure.
    """

    DEFAULT_MAPPINGS = {
        'timestamp': ['Timestamp', 'timestamp', 'date', 'transaction_date'],
        'from_bank': ['From Bank', 'from_bank', 'sender_bank', 'source_bank'],
        'to_bank': ['To Bank', 'to_bank', 'receiver_bank', 'destination_bank'],
        'account': ['Account', 'text_formatAccount', 'account_number', 'account_id'],
        'amount_received': ['Amount Received', 'amount_received', 'received_amount', 'credit_amount'],
        'amount_paid': ['Amount Paid', 'amount_paid', 'paid_amount', 'debit_amount'],
        'receiving_currency': ['text_formatReceiving Currency', 'receiving_currency', 'currency_received',
                               'to_currency'],
        'payment_currency': ['text_formatPayment Currency', 'payment_currency', 'currency_paid', 'from_currency'],
        'payment_format': ['text_formatPayment Format', 'payment_format', 'transaction_type', 'payment_method']
    }

    def __init__(self, custom_mappings: Optional[Dict[str, List[str]]] = None):
        """
        Initialize ColumnMapper with optional custom mappings.

        Args:
            custom_mappings: Dictionary of additional column mappings to merge with defaults
        """
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        if custom_mappings:
            for key, values in custom_mappings.items():
                if key in self.mappings:
                    self.mappings[key].extend(values)
                else:
                    self.mappings[key] = values

    def find_column(self, df: pd.DataFrame, standard_name: str) -> Optional[str]:
        """
        Find the actual column name in DataFrame for a given standard name.

        Args:
            df: Input DataFrame
            standard_name: Standard column name to look for

        Returns:
            Actual column name if found, None otherwise
        """
        if standard_name not in self.mappings:
            return None

        for possible_name in self.mappings[standard_name]:
            if possible_name in df.columns:
                return possible_name
        return None

    def validate_columns(self, df: pd.DataFrame) -> Tuple[bool, str, Dict[str, str]]:
        """
        Validate DataFrame columns and return column mapping.

        Args:
            df: Input DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message, column_mapping)
        """
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
    Preprocesses transaction data for fraud detection by handling feature engineering,
    scaling, and dimensionality reduction.
    """

    def __init__(self,
                 custom_mappings: Optional[Dict[str, List[str]]] = None,
                 pca_components: float = 0.95,
                 random_state: int = 42):
        """
        Initialize preprocessor with configuration options.

        Args:
            custom_mappings: Optional custom column mappings
            pca_components: PCA variance ratio to preserve
            random_state: Random seed for reproducibility
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=random_state)
        self.column_mapper = ColumnMapper(custom_mappings)
        self.column_mapping = None
        self.feature_names = None
        self.fitted = False

    def _process_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp column.

        Args:
            df: Input DataFrame with timestamp column

        Returns:
            DataFrame with temporal features
        """
        timestamp_col = self.column_mapping['timestamp']
        try:
            dt = pd.to_datetime(df[timestamp_col])
            features = pd.DataFrame({
                'hour': dt.dt.hour,
                'day': dt.dt.day,
                'month': dt.dt.month,
                'day_of_week': dt.dt.dayofweek,
                'is_weekend': dt.dt.dayofweek.isin([5, 6]).astype(int),
                'is_business_hours': ((dt.dt.hour >= 9) & (dt.dt.hour <= 17)).astype(int)
            })
            return features
        except Exception as e:
            warnings.warn(f"Error processing temporal features: {str(e)}")
            return pd.DataFrame(index=df.index).assign(
                hour=0, day=1, month=1, day_of_week=0,
                is_weekend=0, is_business_hours=0
            )

    def _process_monetary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process and engineer monetary-related features.

        Args:
            df: Input DataFrame with amount columns

        Returns:
            DataFrame with monetary features
        """
        features = pd.DataFrame()

        # Basic amount features
        features['amount_received'] = pd.to_numeric(
            df[self.column_mapping['amount_received']], errors='coerce'
        ).fillna(0)

        features['amount_paid'] = pd.to_numeric(
            df[self.column_mapping['amount_paid']], errors='coerce'
        ).fillna(0)

        # Derived features
        features['amount_difference'] = features['amount_paid'] - features['amount_received']
        features['amount_ratio'] = np.where(
            features['amount_received'] != 0,
            features['amount_paid'] / features['amount_received'],
            0
        )

        # Log transformations for better distribution
        features['log_amount_received'] = np.log1p(features['amount_received'])
        features['log_amount_paid'] = np.log1p(features['amount_paid'])

        return features

    def _process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process categorical columns into dummy variables.

        Args:
            df: Input DataFrame with categorical columns

        Returns:
            DataFrame with one-hot encoded features
        """
        categorical_cols = [
            'from_bank', 'to_bank', 'receiving_currency',
            'payment_currency', 'payment_format'
        ]

        dummies = []
        for col in categorical_cols:
            mapped_col = self.column_mapping[col]
            dummy = pd.get_dummies(df[mapped_col], prefix=col)
            dummies.append(dummy)

        return pd.concat(dummies, axis=1)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessor and transform the data.

        Args:
            df: Input DataFrame to process

        Returns:
            Numpy array of processed features
        """
        # Validate columns
        is_valid, error_msg, column_mapping = self.column_mapper.validate_columns(df)
        if not is_valid:
            raise ValueError(f"Invalid DataFrame structure: {error_msg}")

        self.column_mapping = column_mapping

        # Extract features
        temporal_features = self._process_temporal_features(df)
        monetary_features = self._process_monetary_features(df)
        categorical_features = self._process_categorical_features(df)

        # Combine all features
        features_df = pd.concat(
            [temporal_features, monetary_features, categorical_features],
            axis=1
        ).fillna(0)

        # Save feature names
        self.feature_names = features_df.columns.tolist()

        # Scale features
        scaled_features = self.scaler.fit_transform(features_df)

        # Apply PCA
        reduced_features = self.pca.fit_transform(scaled_features)

        self.fitted = True
        return reduced_features

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.

        Args:
            df: Input DataFrame to transform

        Returns:
            Numpy array of processed features
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Validate columns
        is_valid, error_msg, _ = self.column_mapper.validate_columns(df)
        if not is_valid:
            raise ValueError(f"Invalid DataFrame structure: {error_msg}")

        # Extract features
        temporal_features = self._process_temporal_features(df)
        monetary_features = self._process_monetary_features(df)
        categorical_features = self._process_categorical_features(df)

        # Combine features
        features_df = pd.concat(
            [temporal_features, monetary_features, categorical_features],
            axis=1
        ).fillna(0)

        # Align features with training columns
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[self.feature_names]

        # Apply scaling and PCA
        scaled_features = self.scaler.transform(features_df)
        reduced_features = self.pca.transform(scaled_features)

        return reduced_features