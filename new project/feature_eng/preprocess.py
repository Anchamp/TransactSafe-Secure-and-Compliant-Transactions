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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Optional, Tuple

class ColumnMapper:
    """Handles flexible column mapping and validation"""
    DEFAULT_MAPPINGS = {
        'timestamp': ['Timestamp', '2023-01-01 00:00:00', '2023-01-01 01:00:00'],
        'from_bank': ['From Bank'],
        'to_bank': ['To Bank'],
        'account': ['Account', 'text_formatAccount'],
        'amount_received': ['Amount Received'],
        'amount_paid': ['Amount Paid'],
        'receiving_currency': ['Receiving Currency', 'text_formatReceiving Currency'],
        'payment_currency': ['Payment Currency', 'text_formatPayment Currency'],
        'payment_format': ['Payment Format', 'text_formatPayment Format']
    }

    def __init__(self, custom_mappings: Optional[Dict[str, List[str]]] = None):
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        if custom_mappings:
            self.mappings.update(custom_mappings)

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
    def __init__(self, batch_size: int = 1000, custom_column_mappings: Optional[Dict[str, List[str]]] = None):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.fitted = False
        self.column_mapper = ColumnMapper(custom_column_mappings)
        self.column_mapping: Optional[Dict[str, str]] = None

    def _process_categorical(self, series: pd.Series, encoder_key: str, fit: bool = False) -> np.ndarray:
        """Process categorical data with streaming support"""
        if fit:
            if encoder_key not in self.label_encoders:
                self.label_encoders[encoder_key] = LabelEncoder()
            return self.label_encoders[encoder_key].fit_transform(series)
        else:
            # Handle unknown categories
            known_categories = set(self.label_encoders[encoder_key].classes_)
            series = series.map(lambda x: x if x in known_categories else self.label_encoders[encoder_key].classes_[0])
            return self.label_encoders[encoder_key].transform(series)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data"""
        # Validate and get column mapping
        is_valid, error_msg, self.column_mapping = self.column_mapper.validate_columns(df)
        if not is_valid:
            raise ValueError(f"Invalid DataFrame structure: {error_msg}")

        # Process timestamp
        df['Timestamp'] = pd.to_datetime(df[self.column_mapping['timestamp']])

        # Process bank codes
        for bank_col in ['from_bank', 'to_bank']:
            encoder_key = f"{bank_col}_encoder"
            df[f"{bank_col}_encoded"] = self._process_categorical(
                df[self.column_mapping[bank_col]], encoder_key, fit=True
            )

        # Process account numbers
        df['account_numeric'] = pd.to_numeric(
            df[self.column_mapping['account']].str.extract('(\d+)').iloc[:, 0],
            errors='coerce'
        ).fillna(0)

        # Process amounts
        df['amount_received'] = pd.to_numeric(df[self.column_mapping['amount_received']], errors='coerce').fillna(0)
        df['amount_paid'] = pd.to_numeric(df[self.column_mapping['amount_paid']], errors='coerce').fillna(0)

        # Process currencies and payment format
        for currency_col in ['receiving_currency', 'payment_currency']:
            encoder_key = f"{currency_col}_encoder"
            df[f"{currency_col}_encoded"] = self._process_categorical(
                df[self.column_mapping[currency_col]], encoder_key, fit=True
            )

        df['payment_format_encoded'] = self._process_categorical(
            df[self.column_mapping['payment_format']], 'payment_format_encoder', fit=True
        )

        # Fit scaler
        self.scaler.fit(df[['amount_received', 'amount_paid']])

        # Convert to final feature set
        feature_cols = (['Timestamp'] +
                       [f"{bank_col}_encoded" for bank_col in ['from_bank', 'to_bank']] +
                       ['account_numeric', 'amount_received', 'amount_paid'] +
                       [f"{currency_col}_encoded" for currency_col in ['receiving_currency', 'payment_currency']] +
                       ['payment_format_encoded'])
        X = df[feature_cols]

        self.fitted = True
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Validate columns
        is_valid, error_msg, _ = self.column_mapper.validate_columns(df)
        if not is_valid:
            raise ValueError(f"Invalid DataFrame structure: {error_msg}")

        # Process timestamp
        df['Timestamp'] = pd.to_datetime(df[self.column_mapping['timestamp']])

        # Process bank codes
        for bank_col in ['from_bank', 'to_bank']:
            encoder_key = f"{bank_col}_encoder"
            df[f"{bank_col}_encoded"] = self._process_categorical(
                df[self.column_mapping[bank_col]], encoder_key, fit=False
            )

        # Process account numbers
        df['account_numeric'] = pd.to_numeric(
            df[self.column_mapping['account']].str.extract('(\d+)').iloc[:, 0],
            errors='coerce'
        ).fillna(0)

        # Process amounts
        df['amount_received'] = pd.to_numeric(df[self.column_mapping['amount_received']], errors='coerce').fillna(0)
        df['amount_paid'] = pd.to_numeric(df[self.column_mapping['amount_paid']], errors='coerce').fillna(0)

        # Process currencies and payment format
        for currency_col in ['receiving_currency', 'payment_currency']:
            encoder_key = f"{currency_col}_encoder"
            df[f"{currency_col}_encoded"] = self._process_categorical(
                df[self.column_mapping[currency_col]], encoder_key, fit=False
            )

        df['payment_format_encoded'] = self._process_categorical(
            df[self.column_mapping['payment_format']], 'payment_format_encoder', fit=False
        )

        # Scale amounts
        df[['amount_received', 'amount_paid']] = self.scaler.transform(
            df[['amount_received', 'amount_paid']]
        )

        # Convert to final feature set
        feature_cols = (['Timestamp'] +
                       [f"{bank_col}_encoded" for bank_col in ['from_bank', 'to_bank']] +
                       ['account_numeric', 'amount_received', 'amount_paid'] +
                       [f"{currency_col}_encoded" for currency_col in ['receiving_currency', 'payment_currency']] +
                       ['payment_format_encoded'])
        X = df[feature_cols]

        return X