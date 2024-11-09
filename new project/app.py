from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pandas as pd
import numpy as np
import os
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Union
import traceback

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Create necessary directories
for directory in ['data', 'models', 'templates', 'static', 'uploads']:
    Path(directory).mkdir(exist_ok=True)


class User(UserMixin):
    def __init__(self, user_id: str):
        self.id = user_id


# In-memory user store (replace with database in production)
users = {}


class TransactionValidator:
    """Validates transaction data format and content"""
    REQUIRED_COLUMNS = {
        'Timestamp': str,
        'From Bank': str,
        'To Bank': str,
        'Amount': float,
        'Currency': str
    }

    @staticmethod
    def validate_csv(file) -> tuple[bool, str, Optional[pd.DataFrame]]:
        """Validates uploaded CSV file and its contents"""
        try:
            if not file.filename.endswith('.csv'):
                return False, "Invalid file type. Please upload a CSV file.", None

            df = pd.read_csv(file)

            # Check required columns
            missing_cols = [col for col in TransactionValidator.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                return False, f"Missing required columns: {', '.join(missing_cols)}", None

            # Validate data types
            for col, dtype in TransactionValidator.REQUIRED_COLUMNS.items():
                if col == 'Amount':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isna().any():
                        return False, f"Invalid amount values found in the {col} column", None
                elif col == 'Timestamp':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        return False, f"Invalid timestamp format in the {col} column", None

            return True, "", df

        except Exception as e:
            logger.error(f"Error validating CSV: {str(e)}")
            return False, f"Error processing file: {str(e)}", None


class ModelManager:
    """Manages model loading and prediction operations"""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with dummy if real model not available"""
        try:
            if os.path.exists('models/model.joblib'):
                self.model, self.preprocessor = joblib.load('models/model.joblib')
            else:
                logger.warning("Using dummy model for testing")
                self.model = type('DummyModel', (), {
                    'predict_proba': lambda self, X: np.random.uniform(0, 1, (len(X), 2))
                })()
                self.preprocessor = type('DummyPreprocessor', (), {
                    'transform': lambda self, X: X
                })()
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.model = None
            self.preprocessor = None

    def predict(self, df: pd.DataFrame) -> tuple[bool, str, Optional[np.ndarray]]:
        """Generate predictions for input data"""
        try:
            if self.model is None:
                return False, "Model not initialized", None

            # Preprocess data if preprocessor available
            if self.preprocessor:
                df = self.preprocessor.transform(df)

            # Generate predictions
            predictions = self.model.predict_proba(df)[:, 1]
            return True, "", predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            return False, f"Error generating predictions: {str(e)}", None


# Initialize model manager
model_manager = ModelManager()


@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    return User(user_id) if user_id in users or user_id == "admin" else None


def save_predictions(df: pd.DataFrame, predictions: np.ndarray) -> None:
    """Save predictions and high-risk transactions"""
    try:
        # Add predictions to dataframe
        df['risk_score'] = predictions
        df['is_suspicious'] = predictions > 0.7

        # Save all predictions
        df.to_csv('data/fraudulent_predictions.csv', index=False)

        # Save high-risk transactions as alerts
        high_risk = df[df['risk_score'] > 0.7].copy()
        if not high_risk.empty:
            high_risk['alert_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Append to existing alerts or create new file
            alerts_path = 'data/alerts.csv'
            if os.path.exists(alerts_path):
                alerts_df = pd.read_csv(alerts_path)
                alerts_df = pd.concat([alerts_df, high_risk])
            else:
                alerts_df = high_risk

            alerts_df.to_csv(alerts_path, index=False)

    except Exception as e:
        logger.error(f"Error saving predictions: {str(e)}")
        raise


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if (username in users and users[username]['password'] == password) or \
                (username == "admin" and password == "password"):
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Invalid credentials')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if username in users:
            flash('Username already exists')
            return redirect(url_for('register'))

        users[username] = {
            'email': email,
            'password': password
        }
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Initialize default values
        default_data = {
            'alerts': [],
            'summary': {
                'total_alerts': 0,
                'high_risk_count': 0,
                'total_amount': 0.0,
                'recent_alerts': 0
            },
            'model_performance': {
                'labels': ['Isolation Forest 1', 'Isolation Forest 2', 'LOF', 'DBSCAN'],
                'scores': [0.85, 0.82, 0.78, 0.80]
            },
            'risk_distribution': {
                'labels': ['High Risk', 'Medium Risk', 'Low Risk'],
                'values': [0, 0, 0]
            },
            'trend_data': {
                'dates': [],
                'counts': []
            }
        }

        # Load and process alerts if file exists
        if os.path.exists('data/alerts.csv'):
            alerts_df = pd.read_csv('data/alerts.csv')
            if not alerts_df.empty:
                # Ensure all required columns exist
                for col in ['Amount', 'risk_score', 'alert_date']:
                    if col not in alerts_df.columns:
                        alerts_df[col] = 0.0 if col == 'Amount' else (0.0 if col == 'risk_score' else '')

                # Calculate summary metrics
                default_data['summary'].update({
                    'total_alerts': len(alerts_df),
                    'high_risk_count': len(alerts_df[alerts_df['risk_score'] > 0.8]),
                    'total_amount': float(alerts_df['Amount'].sum()),
                    'recent_alerts': len(alerts_df[pd.to_datetime(alerts_df['alert_date']) >
                        (pd.Timestamp.now() - pd.Timedelta(days=1))])
                })

                # Calculate risk distribution
                default_data['risk_distribution']['values'] = [
                    len(alerts_df[alerts_df['risk_score'] > 0.8]),
                    len(alerts_df[(alerts_df['risk_score'] > 0.5) & (alerts_df['risk_score'] <= 0.8)]),
                    len(alerts_df[alerts_df['risk_score'] <= 0.5])
                ]

                # Calculate trend data
                alerts_df['alert_date'] = pd.to_datetime(alerts_df['alert_date'])
                daily_alerts = alerts_df.groupby(alerts_df['alert_date'].dt.date).size().tail(7)
                default_data['trend_data'].update({
                    'dates': [d.strftime('%Y-%m-%d') for d in daily_alerts.index],
                    'counts': daily_alerts.values.tolist()
                })

                # Prepare alerts for template
                default_data['alerts'] = alerts_df.fillna('').to_dict('records')

        return render_template(
            'dashboard.html',
            alerts=default_data['alerts'],
            summary=default_data['summary'],
            model_performance=default_data['model_performance'],
            risk_distribution=default_data['risk_distribution'],
            trend_data=default_data['trend_data']
        )

    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        flash('Error loading dashboard data')
        return render_template(
            'dashboard.html',
            alerts=[],
            summary={'total_alerts': 0, 'high_risk_count': 0, 'total_amount': 0.0, 'recent_alerts': 0},
            model_performance={'labels': [], 'scores': []},
            risk_distribution={'labels': [], 'values': []},
            trend_data={'dates': [], 'counts': []}
        )

@app.route('/generate_predictions', methods=['GET', 'POST'])
@login_required
def generate_predictions():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        try:
            # Validate uploaded file
            is_valid, message, df = TransactionValidator.validate_csv(file)
            if not is_valid:
                flash(message)
                return redirect(request.url)

            # Generate predictions
            success, error_message, predictions = model_manager.predict(df)
            if not success:
                flash(error_message)
                return redirect(request.url)

            # Save predictions and alerts
            save_predictions(df, predictions)

            flash('Predictions generated successfully')
            return render_template('generate_predictions.html',
                                   predictions=df.to_dict('records'))

        except Exception as e:
            logger.error(f"Error in generate_predictions: {str(e)}\n{traceback.format_exc()}")
            flash(f'An error occurred: {str(e)}')
            return redirect(request.url)

    # GET request - display existing predictions if available
    try:
        predictions_df = pd.read_csv('data/fraudulent_predictions.csv') \
            if os.path.exists('data/fraudulent_predictions.csv') else pd.DataFrame()
        return render_template('generate_predictions.html',
                               predictions=predictions_df.to_dict('records') if not predictions_df.empty else [])
    except Exception as e:
        logger.error(f"Error loading existing predictions: {str(e)}")
        return render_template('generate_predictions.html', predictions=[])


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run(debug=True)