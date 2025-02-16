
## TransactSafe –Secure and Compliant Transactions
A robust Anti-Money Laundering (AML) transaction monitoring system built with Flask, machine learning, and HBase. The system provides real-time analysis of financial transactions to detect suspicious patterns and potential money laundering activities.

## Features
- Real-time transaction monitoring and risk scoring
- Unsupervised machine learning for anomaly detection
- Interactive dashboard for monitoring alerts and trends
- Batch processing capability for large transaction files
- Secure user authentication and role-based access
- HBase integration for scalable data storage
- Comprehensive feature engineering and preprocessing
- Multi-model ensemble approach for improved accuracy

## Tech Stack
- **Backend**: Python, Flask
- **Database**: HBase
- **Machine Learning**: scikit-learn, NumPy, Pandas
- **Frontend**: HTML, CSS, JavaScript
- **Message Queue**: Apache Kafka
- **Containerization**: Docker
##Requirements

-Python 3.8+
-HBase 2.x
-Apache Kafka
-Docker and Docker Compose
-Required Python packages listed in requirements.txt
# AML Transaction Monitoring System



## Project Structure
```
aml-transaction-monitoring/
├── app/
│   ├── core/              # Core functionality and models
│   ├── database/          # Database operations
│   ├── ml/               # Machine learning components
│   ├── api/              # API endpoints
│   └── utils/            # Utility functions
├── docker/               # Docker configuration
├── tests/               # Test suite
├── static/              # Static assets
├── templates/           # HTML templates
└── scripts/             # Utility scripts
```


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aml-transaction-monitoring.git
cd aml-transaction-monitoring
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up HBase:
```bash
docker-compose up -d hbase
python scripts/init_hbase.py
```

5. Initialize the application:
```bash
python app/main.py
```

## Configuration
1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Update the following configurations in `.env`:
```
SECRET_KEY=your-secret-key
HBASE_HOST=localhost
HBASE_PORT=9090
MODEL_PATH=models/aml_detector.joblib
```

## Usage

### Starting the Application
1. Start the services:
```bash
docker-compose up -d
```

2. Run the Flask application:
```bash
python app/main.py
```

3. Access the application at `http://localhost:5000`

### Processing Transactions
1. Log in to the application
2. Navigate to "Generate Predictions"
3. Upload a CSV file containing transactions
4. View results in the dashboard

### Monitoring
- View real-time alerts in the dashboard
- Monitor processing status for batch uploads
- Export reports and analytics

## Machine Learning Components

### Feature Engineering
- Temporal patterns analysis
- Amount-based features
- Bank relationship features
- Categorical encoding
- Interaction features

### Model Ensemble
- Isolation Forest
- Local Outlier Factor (LOF)
- DBSCAN
- Weighted voting system


