from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
import logging
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def get_model_path(self, stock_symbol: str) -> str:
        return os.path.join(self.model_dir, f'{stock_symbol}_model.h5')
    
    def get_scaler_path(self, stock_symbol: str) -> str:
        return os.path.join(self.model_dir, f'{stock_symbol}_scaler.pkl')
    
    def save_model(self, model: Sequential, scaler: MinMaxScaler, stock_symbol: str) -> None:
        model.save(self.get_model_path(stock_symbol))
        joblib.dump(scaler, self.get_scaler_path(stock_symbol))
        
    def load_model(self, stock_symbol: str):
        model_path = self.get_model_path(stock_symbol)
        scaler_path = self.get_scaler_path(stock_symbol)
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        return None, None

class StockDataFetcher:
    @staticmethod
    def fetch_historical_data(stock_symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(stock_symbol)
            hist = stock.history(start=start_date, end=end_date)
            logger.info(f"Fetched historical data for {stock_symbol}: {hist.shape}")
            
            if hist.empty:
                raise ValueError(f"No data found for {stock_symbol}")
                
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    @staticmethod
    def fetch_live_data(stock_symbol: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(stock_symbol)
            live_data = stock.history(period='1d', interval='1m')
            logger.info(f"Fetched live data for {stock_symbol}: {live_data.shape}")
            return live_data
        except Exception as e:
            logger.error(f"Error fetching live data: {str(e)}")
            raise

class DataPreprocessor:
    def __init__(self, time_step: int = 10):
        self.time_step = time_step
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        data = data[['Close']].dropna()
        scaled_data = self.scaler.fit_transform(data)
        logger.info(f"Preprocessed data shape: {scaled_data.shape}")
        return scaled_data
        
    def create_dataset(self, data: np.ndarray):
        X, Y = [], []
        for i in range(len(data) - self.time_step - 1):
            X.append(data[i:(i + self.time_step), 0])
            Y.append(data[i + self.time_step, 0])
            
        X = np.array(X)
        Y = np.array(Y)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        logger.info(f"Created dataset with {len(X)} samples")
        return X, Y

class ModelBuilder:
    @staticmethod
    def build_model(input_shape) -> Sequential:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    @staticmethod
    def predict_future(model: Sequential, data: np.ndarray, time_step: int, 
                      days_to_predict: int, scaler: MinMaxScaler) -> list:
        predictions = []
        current_data = data[-time_step:]
        
        for _ in range(days_to_predict):
            current_data_reshaped = current_data.reshape(1, time_step, 1)
            prediction = model.predict(current_data_reshaped, verbose=0)
            
            # Inverse transform the prediction
            pred_array = np.zeros((1, 1))
            pred_array[0, 0] = prediction[0, 0]
            prediction_original = scaler.inverse_transform(pred_array)
            predictions.append(float(prediction_original[0, 0]))
            
            # Update current_data for next prediction
            current_data = np.append(current_data[1:], prediction, axis=0)
            
        logger.info(f"Generated {len(predictions)} predictions")
        return predictions

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Input validation
        stock_symbol = request.args.get('stock')
        days_to_predict = request.args.get('days')
        
        if not stock_symbol:
            return jsonify({"error": "Stock symbol is required"}), 400
            
        try:
            days_to_predict = int(days_to_predict)
            if days_to_predict <= 0 or days_to_predict > 30:
                return jsonify({"error": "Days to predict must be between 1 and 30"}), 400
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid days parameter"}), 400

        # Initialize components
        model_manager = ModelManager()
        data_fetcher = StockDataFetcher()
        data_preprocessor = DataPreprocessor()
        
        try:
            # Fetch and prepare data
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            historical_data = data_fetcher.fetch_historical_data(stock_symbol, start_date, end_date)
            live_data = data_fetcher.fetch_live_data(stock_symbol)
            combined_data = pd.concat([historical_data, live_data])
            
            # Get or create model
            model, scaler = model_manager.load_model(stock_symbol)
            
            if model is None or scaler is None:
                # Preprocess data
                processed_data = data_preprocessor.preprocess_data(combined_data)
                X, y = data_preprocessor.create_dataset(processed_data)
                
                # Build and train model
                model = ModelBuilder.build_model((data_preprocessor.time_step, 1))
                model.fit(X, y, epochs=50, batch_size=32, verbose=0)
                
                # Save model and scaler
                model_manager.save_model(model, data_preprocessor.scaler, stock_symbol)
                scaler = data_preprocessor.scaler
            
            # Generate predictions
            predictions = ModelBuilder.predict_future(
                model, 
                scaler.transform(combined_data[['Close']]),
                data_preprocessor.time_step,
                days_to_predict,
                scaler
            )
            
            # Prepare response
            dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                    for i in range(1, days_to_predict + 1)]
                    
            response = {
                "stock_symbol": stock_symbol,
                "predictions": [
                    {"date": date, "price": float(price)} 
                    for date, price in zip(dates, predictions)
                ]
            }
            
            return jsonify(response)
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 500
            
    except Exception as e:
        error_msg = f"Error in prediction endpoint: {str(e)}"
        logger.error(error_msg)
        return jsonify({"error": error_msg}), 500

# if __name__ == '__main__':
#     # Create necessary directories
#     os.makedirs('models', exist_ok=True)
#     os.makedirs('templates', exist_ok=True)
    
#     # Start the Flask app
#     app.run(port=3000, debug=True)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Define directories to ignore
    ignore_dirs = [
        os.path.join(os.path.dirname(__file__), 'venv'),
        os.path.join(os.path.dirname(__file__), 'Lib'),
        os.path.join(os.path.dirname(__file__), 'site-packages')
    ]

    # Run the Flask app with reloader ignoring certain directories
    app.run(port=3000, debug=True, extra_files=ignore_dirs)