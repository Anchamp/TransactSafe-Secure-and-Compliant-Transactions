import os


class Config:
    # Get the base project directory
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    # Data paths
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'HI-Medium_Trans.csv')
    ALERTS_PATH = os.path.join(DATA_DIR, 'alerts.csv')
    PREDICTIONS_PATH = os.path.join(DATA_DIR, 'fraudulent_predictions.csv')

    # Model paths
    MODEL_DIR = os.path.join(BASE_DIR, 'model')
    MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')

    # Flask config
    SECRET_KEY = 'your-secret-key-here'  # Change this in production

    @staticmethod
    def init_directories():
        """Create necessary directories if they don't exist"""
        for directory in [Config.DATA_DIR, Config.MODEL_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory)

