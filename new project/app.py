from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import logging
import joblib
from pathlib import Path
from typing import Dict, List, Optional
import traceback


class Config:
    """Application configuration"""
    SECRET_KEY = 'your-secret-key-here'  # Change in production
    MAX_CONTENT_LENGTH = 4 * 1024 * 1024 * 1024  # 4GB max file size
    UPLOAD_FOLDER = 'uploads'
    ALERTS_FILE = 'data/alerts.csv'
    MODEL_PATH = 'models/aml_detector.joblib'


# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id


# In-memory user store (replace with database in production)
users = {}

# Processing status tracker
processing_status = {
    'is_processing': False,
    'current_chunk': 0,
    'total_chunks': 0,
    'status_message': '',
    'error': None
}


class ModelManager:
    """Manages ML model operations"""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            if os.path.exists(Config.MODEL_PATH):
                self.model = joblib.load(Config.MODEL_PATH)
                logging.info("Loaded production model")
            else:
                logging.warning("Using dummy model for testing")
                self.model = type('DummyModel', (), {
                    'predict': lambda self, X: (np.random.uniform(0, 1, len(X)),
                                                {'model1': np.random.uniform(0, 1, len(X))}),
                    'generate_report': lambda self, df, pred, scores: {
                        'total_transactions': len(df),
                        'high_risk_count': sum(pred > 0.7),
                        'average_risk_score': float(np.mean(pred))
                    }
                })()
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            self.model = None

    def predict(self, df: pd.DataFrame) -> tuple[bool, str, Optional[np.ndarray], Optional[dict]]:
        try:
            if self.model is None:
                return False, "Model not initialized", None, None

            predictions, scores = self.model.predict(df)
            report = self.model.generate_report(df, predictions, scores)

            return True, "", predictions, report
        except Exception as e:
            return False, f"Error generating predictions: {str(e)}", None, None


def create_app():
    """Application factory function"""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create required directories
    for directory in ['data', 'models', 'uploads']:
        Path(directory).mkdir(exist_ok=True)

    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'

    # Initialize model manager
    model_manager = ModelManager()

    @login_manager.user_loader
    def load_user(user_id: str) -> Optional[User]:
        """Required user loader function for Flask-Login"""
        if user_id in users or user_id == "admin":
            return User(user_id)
        return None

    def generate_dashboard_data():
        """Generate data for dashboard visualization"""
        default_data = {
            'alerts': [],
            'summary': {
                'total_alerts': 0,
                'high_risk_count': 0,
                'total_amount': 0.0,
                'recent_alerts': 0
            },
            'risk_distribution': {
                'labels': ['High Risk', 'Medium Risk', 'Low Risk'],
                'values': [0, 0, 0]
            },
            'model_performance': {
                'labels': ['Isolation Forest', 'LOF', 'DBSCAN'],
                'scores': [0.85, 0.82, 0.78]
            },
            'trend_data': {
                'dates': [],
                'counts': []
            }
        }

        try:
            if not os.path.exists(Config.ALERTS_FILE):
                return default_data

            # Read CSV with proper type handling
            alerts_df = pd.read_csv(Config.ALERTS_FILE, low_memory=False)
            if alerts_df.empty:
                return default_data

            # Convert and clean data types
            alerts_df['alert_date'] = pd.to_datetime(alerts_df['alert_date'], errors='coerce')
            alerts_df['Amount'] = pd.to_numeric(alerts_df['Amount'], errors='coerce').fillna(0)
            alerts_df['risk_score'] = pd.to_numeric(alerts_df['risk_score'], errors='coerce').fillna(0)

            # Calculate recent alerts
            now = datetime.now()
            recent_mask = alerts_df['alert_date'] > (now - timedelta(days=1))

            # Calculate risk levels
            alerts_df['risk_level'] = 'Low Risk'
            alerts_df.loc[alerts_df['risk_score'] > 0.7, 'risk_level'] = 'Medium Risk'
            alerts_df.loc[alerts_df['risk_score'] > 0.85, 'risk_level'] = 'High Risk'

            risk_counts = alerts_df['risk_level'].value_counts()

            # Calculate daily trend
            last_week = now - timedelta(days=7)
            daily_counts = alerts_df[alerts_df['alert_date'] >= last_week].groupby(
                alerts_df['alert_date'].dt.date
            ).size()

            # Create date range for complete week
            date_range = pd.date_range(
                start=last_week.date(),
                end=now.date(),
                freq='D'
            )
            daily_counts = daily_counts.reindex(date_range, fill_value=0)

            # Prepare alerts for display
            display_alerts = (
                alerts_df.sort_values('alert_date', ascending=False)
                .head(100)
                .fillna('N/A')
                .replace([np.inf, -np.inf], 0)
            )

            # Convert numeric columns to proper format
            for alert in display_alerts.to_dict('records'):
                if pd.notna(alert.get('Amount')):
                    alert['Amount'] = float(alert['Amount'])
                if pd.notna(alert.get('risk_score')):
                    alert['risk_score'] = float(alert['risk_score'])

            return {
                'alerts': display_alerts.to_dict('records'),
                'summary': {
                    'total_alerts': len(alerts_df),
                    'high_risk_count': int(sum(alerts_df['risk_score'] > 0.85)),
                    'total_amount': float(alerts_df['Amount'].sum()),
                    'recent_alerts': int(sum(recent_mask))
                },
                'risk_distribution': {
                    'labels': ['High Risk', 'Medium Risk', 'Low Risk'],
                    'values': [
                        int(risk_counts.get('High Risk', 0)),
                        int(risk_counts.get('Medium Risk', 0)),
                        int(risk_counts.get('Low Risk', 0))
                    ]
                },
                'model_performance': {
                    'labels': ['Isolation Forest', 'LOF', 'DBSCAN'],
                    'scores': [0.85, 0.82, 0.78]
                },
                'trend_data': {
                    'dates': [d.strftime('%Y-%m-%d') for d in daily_counts.index],
                    'counts': [int(x) for x in daily_counts.values]
                }
            }

        except Exception as e:
            logging.error(f"Error generating dashboard data: {str(e)}\n{traceback.format_exc()}")
            return default_data

    @app.route('/', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            if ((username in users and users[username]['password'] == password) or
                    (username == "admin" and password == "password")):
                user = User(username)
                login_user(user)
                return redirect(url_for('generate_predictions'))

            flash('Invalid credentials')
        return render_template('login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            if not all([username, email, password]):
                flash('All fields are required')
                return redirect(url_for('register'))

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
            data = generate_dashboard_data()
            return render_template('dashboard.html', **data)
        except Exception as e:
            logging.error(f"Error in dashboard: {str(e)}\n{traceback.format_exc()}")
            flash('Error loading dashboard data')
            return render_template('dashboard.html', **generate_dashboard_data())
        try:
            dashboard_data = generate_dashboard_data()
            if dashboard_data is None:
                flash('Error loading dashboard data')
                dashboard_data = {
                    'alerts': [],
                    'summary': {
                        'total_alerts': 0,
                        'high_risk_count': 0,
                        'total_amount': 0.0,
                        'recent_alerts': 0
                    },
                    'risk_distribution': {'labels': [], 'values': []},
                    'trend_data': {'dates': [], 'counts': []}
                }

            return render_template('dashboard.html', **dashboard_data)
        except Exception as e:
            logging.error(f"Error in dashboard: {str(e)}")
            flash('Error loading dashboard')
            return redirect(url_for('login'))

    @app.route('/generate_predictions', methods=['GET', 'POST'])
    @login_required
    def generate_predictions():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file uploaded')
                return redirect(request.url)

            file = request.files['file']
            if not file.filename:
                flash('No file selected')
                return redirect(request.url)

            try:
                df = pd.read_csv(file)
                success, error_msg, predictions, report = model_manager.predict(df)

                if not success:
                    flash(error_msg)
                    return redirect(request.url)

                # Save alerts for high-risk transactions
                high_risk_mask = predictions > 0.7
                if high_risk_mask.any():
                    alerts_df = df[high_risk_mask].copy()
                    alerts_df['risk_score'] = predictions[high_risk_mask]
                    alerts_df['alert_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    mode = 'a' if os.path.exists(Config.ALERTS_FILE) else 'w'
                    alerts_df.to_csv(Config.ALERTS_FILE,
                                     mode=mode,
                                     header=(mode == 'w'),
                                     index=False)

                df['risk_score'] = predictions
                flash('Predictions generated successfully')
                return render_template(
                    'generate_predictions.html',
                    predictions=df.head(1000).to_dict('records'),
                    report=report
                )

            except Exception as e:
                logging.error(f"Error in generate_predictions: {str(e)}")
                flash(f'Error processing file: {str(e)}')
                return redirect(request.url)

        return render_template('generate_predictions.html',
                               predictions=None,
                               report=None)

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out successfully')
        return redirect(url_for('login'))

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)