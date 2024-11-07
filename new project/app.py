'''# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pandas as pd
import os
import logging
import joblib
from config.config import Config
from feature_eng.preprocess import ColumnMapper, UnsupervisedTransactionPreprocessor

app = Flask(__name__)
app.config.from_object(Config)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize model
def init_model():
    if not os.path.exists(Config.MODEL_PATH):
        logger.error(f"Model file not found at {Config.MODEL_PATH}")
        return None, None
    try:
        model, preprocessor = joblib.load(Config.MODEL_PATH)
        logger.info("Model loaded successfully")
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize model and preprocessor
model, preprocessor = init_model()
'''# app.py
# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pandas as pd
import os
import logging
import joblib
from config.config import Config
from feature_eng.preprocess import ColumnMapper, TransactionPreprocessor

app = Flask(__name__)
app.config.from_object(Config)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize model
def init_model():
    if not os.path.exists(Config.MODEL_PATH):
        logger.error(f"Model file not found at {Config.MODEL_PATH}")
        return None, None
    try:
        model, preprocessor = joblib.load(Config.MODEL_PATH)
        logger.info("Model loaded successfully")
        return model, preprocessor
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None, None

# Initialize model and preprocessor
model, preprocessor = init_model()

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Simple authentication for demo
        if username == "admin" and password == "password":
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

        # Check if the username already exists
        if username in users:
            flash('Username already exists. Please choose a different username.')
            return redirect(url_for('register'))

        # Store the user credentials in the in-memory database
        users[username] = {
            'email': email,
            'password': password
        }

        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the username and password match the stored credentials
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Invalid credentials')

    return render_template('login.html')
@app.route('/dashboard')
@login_required
def dashboard():
    try:
        # Load recent alerts from storage
        alerts_df = pd.read_csv('data/alerts.csv')
        alerts = alerts_df.to_dict('records')
        return render_template('dashboard.html', alerts=alerts)
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        return render_template('dashboard.html', alerts=[])

@app.route('/analyze_transaction', methods=['POST'])
@login_required
def analyze_transaction():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not initialized'})

    try:
        # Get transaction data from form
        transaction_data = {
            'Timestamp': request.form.get('timestamp'),
            'grid_3x3From Bank': request.form.get('from_bank'),
            'text_formatAccount': request.form.get('from_account'),
            'grid_3x3To Bank': request.form.get('to_bank'),
            'text_formatAccount.1': request.form.get('to_account'),
            'grid_3x3Amount Received': float(request.form.get('amount_received')),
            'text_formatReceiving Currency': request.form.get('receiving_currency'),
            'grid_3x3Amount Paid': float(request.form.get('amount_paid')),
            'text_formatPayment Currency': request.form.get('payment_currency'),
            'text_formatPayment Format': request.form.get('payment_format')
        }

        # Convert to DataFrame
        df = pd.DataFrame([transaction_data])

        # Preprocess transaction
        processed_data = preprocessor.transform(df)

        # Make prediction
        risk_score = model.predict_proba(processed_data)[0][1]

        # Save if high risk
        if risk_score > 0.7:
            transaction_data['risk_score'] = risk_score
            transaction_data['alert_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            alerts_df = pd.read_csv('data/alerts.csv')
            alerts_df = pd.concat([alerts_df, pd.DataFrame([transaction_data])])
            alerts_df.to_csv('data/alerts.csv', index=False)

        return jsonify({
            'success': True,
            'risk_score': float(risk_score),
            'is_suspicious': bool(risk_score > 0.7)
        })

    except Exception as e:
        logger.error(f"Error analyzing transaction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/generate_predictions', methods=['GET', 'POST'])
@login_required
def generate_predictions():
    if model is None:
        flash('Model not initialized')
        return render_template('generate_predictions.html', predictions=[])

    if request.method == 'POST':
        try:
            # Load and process batch transactions
            file = request.files['file']
            df = pd.read_csv(file)

            # Preprocess data
            processed_data = preprocessor.transform(df)

            # Generate predictions
            predictions = model.predict_proba(processed_data)[:, 1]

            # Add predictions to dataframe
            df['risk_score'] = predictions
            df['is_suspicious'] = predictions > 0.7

            # Save results
            output_path = 'fraudulent_predictions.csv'
            df.to_csv(output_path, index=False)

            return redirect(url_for('generate_predictions'))

        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            flash('Error processing file')
            return redirect(url_for('generate_predictions'))

    # Display existing predictions
    try:
        predictions_df = pd.read_csv('fraudulent_predictions.csv')
        return render_template('generate_predictions.html',
                               predictions=predictions_df.to_dict('records'))
    except:
        return render_template('generate_predictions.html', predictions=[])


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)