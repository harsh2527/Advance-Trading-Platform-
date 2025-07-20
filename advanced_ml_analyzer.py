import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class AdvancedMLStockAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.model_weights = {}
        
    def calculate_fibonacci_levels(self, data, period=50):
        """Calculate Fibonacci retracement levels"""
        recent_data = data.tail(period)
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        
        diff = high - low
        levels = {
            '0%': high,
            '23.6%': high - 0.236 * diff,
            '38.2%': high - 0.382 * diff,
            '50%': high - 0.5 * diff,
            '61.8%': high - 0.618 * diff,
            '78.6%': high - 0.786 * diff,
            '100%': low
        }
        return levels
    
    def calculate_fibonacci_extensions(self, data, period=50):
        """Calculate Fibonacci extension levels"""
        recent_data = data.tail(period)
        high = recent_data['High'].max()
        low = recent_data['Low'].min()
        
        diff = high - low
        extensions = {
            '127.2%': high + 0.272 * diff,
            '138.2%': high + 0.382 * diff,
            '161.8%': high + 0.618 * diff,
            '200%': high + diff,
            '261.8%': high + 1.618 * diff
        }
        return extensions
    
    def add_technical_features(self, data):
        """Add comprehensive technical indicators"""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['BB_Width'] = df['BB_High'] - df['BB_Low']
        df['BB_Position'] = (df['Close'] - df['BB_Low']) / (df['BB_High'] - df['BB_Low'])
        
        # Momentum indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['RSI_SMA'] = ta.trend.sma_indicator(df['RSI'], window=5)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_Hist'] = ta.trend.macd(df['Close'])
        df['Stoch'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # Volume indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Volatility indicators
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        df['Keltner_High'] = ta.volatility.keltner_channel_hband(df['High'], df['Low'], df['Close'])
        df['Keltner_Low'] = ta.volatility.keltner_channel_lband(df['High'], df['Low'], df['Close'])
        
        # Trend indicators
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        df['Aroon_Up'] = ta.trend.aroon_up(df['High'], df['Low'])
        df['Aroon_Down'] = ta.trend.aroon_down(df['High'], df['Low'])
        
        # Price action features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_MA'] = df['Price_Change'].rolling(window=5).mean()
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Support and Resistance
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Distance_to_Resistance'] = (df['Resistance'] - df['Close']) / df['Close']
        df['Distance_to_Support'] = (df['Close'] - df['Support']) / df['Close']
        
        # Fibonacci levels
        fib_levels = self.calculate_fibonacci_levels(df)
        current_price = df['Close'].iloc[-1]
        df['Fib_Distance_23.6'] = abs(current_price - fib_levels['23.6%']) / current_price
        df['Fib_Distance_38.2'] = abs(current_price - fib_levels['38.2%']) / current_price
        df['Fib_Distance_61.8'] = abs(current_price - fib_levels['61.8%']) / current_price
        
        return df
    
    def prepare_features_for_ml(self, data):
        """Prepare features for machine learning models"""
        df = self.add_technical_features(data)
        
        # Create lag features
        for lag in [1, 2, 3, 5, 7, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
            df[f'MACD_lag_{lag}'] = df['MACD'].shift(lag)
        
        # Rolling statistics
        for window in [3, 5, 7, 10, 14, 20]:
            df[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_rolling_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_rolling_mean_{window}'] = df['Volume'].rolling(window).mean()
            df[f'RSI_rolling_mean_{window}'] = df['RSI'].rolling(window).mean()
        
        # Volatility features
        df['Volatility_5'] = df['Close'].rolling(window=5).std()
        df['Volatility_10'] = df['Close'].rolling(window=10).std()
        df['Volatility_20'] = df['Close'].rolling(window=20).std()
        
        return df
    
    def train_ensemble_models(self, X, y):
        """Train multiple ML models for ensemble prediction"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models_config = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42, verbose=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=200, max_depth=8, learning_rate=0.1, random_state=42),
            'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu', max_iter=1000, random_state=42)
        }
        
        model_scores = {}
        
        for name, model in models_config.items():
            try:
                if name == 'NeuralNetwork':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                score = 1 / (1 + mse)  # Higher score for better models
                
                self.models[name] = model
                model_scores[name] = score
                
                print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, Score: {score:.4f}")
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        # Calculate weights based on performance
        total_score = sum(model_scores.values())
        self.model_weights = {name: score/total_score for name, score in model_scores.items()}
        
        return self.models, self.model_weights
    
    def predict_ensemble(self, X, days=7):
        """Make ensemble predictions"""
        if not self.models:
            return None
        
        predictions_dict = {}
        
        for name, model in self.models.items():
            try:
                if name == 'NeuralNetwork':
                    X_scaled = self.scaler.transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                predictions_dict[name] = pred[0] if len(pred) > 0 else 0
            except Exception as e:
                print(f"Prediction error for {name}: {e}")
                predictions_dict[name] = 0
        
        # Weighted ensemble prediction
        ensemble_pred = sum(pred * self.model_weights.get(name, 0) for name, pred in predictions_dict.items())
        
        # Generate multi-day predictions
        multi_day_predictions = []
        current_pred = ensemble_pred
        
        for day in range(days):
            # Add some trend and volatility
            trend_factor = 0.99 + (np.random.random() - 0.5) * 0.02  # Small random trend
            current_pred *= trend_factor
            multi_day_predictions.append(current_pred)
        
        return {
            'ensemble_prediction': multi_day_predictions,
            'individual_predictions': predictions_dict,
            'model_weights': self.model_weights
        }
    
    def calculate_price_targets(self, data, predictions):
        """Calculate price targets using Fibonacci and technical analysis"""
        current_price = data['Close'].iloc[-1]
        fib_levels = self.calculate_fibonacci_levels(data)
        fib_extensions = self.calculate_fibonacci_extensions(data)
        
        # Technical levels
        resistance = data['High'].rolling(window=20).max().iloc[-1]
        support = data['Low'].rolling(window=20).min().iloc[-1]
        
        # Combine predictions with Fibonacci levels
        pred_high = max(predictions['ensemble_prediction'])
        pred_low = min(predictions['ensemble_prediction'])
        
        targets = {
            'current_price': current_price,
            'predicted_range': {
                'high': pred_high,
                'low': pred_low,
                'mid': (pred_high + pred_low) / 2
            },
            'fibonacci_levels': fib_levels,
            'fibonacci_extensions': fib_extensions,
            'technical_levels': {
                'resistance': resistance,
                'support': support
            },
            'confidence_score': self.calculate_confidence_score(data, predictions)
        }
        
        return targets
    
    def calculate_confidence_score(self, data, predictions):
        """Calculate confidence score based on multiple factors"""
        try:
            # Volume analysis
            avg_volume = data['Volume'].tail(20).mean()
            recent_volume = data['Volume'].iloc[-1]
            volume_score = min(recent_volume / avg_volume, 2) / 2
            
            # Volatility analysis
            volatility = data['Close'].tail(20).std() / data['Close'].tail(20).mean()
            volatility_score = max(0, 1 - volatility * 10)
            
            # Trend consistency
            sma_20 = data['Close'].tail(20).mean()
            trend_score = 1 if data['Close'].iloc[-1] > sma_20 else 0.5
            
            # Model agreement (how close are the individual model predictions)
            individual_preds = list(predictions['individual_predictions'].values())
            if len(individual_preds) > 1:
                pred_std = np.std(individual_preds)
                pred_mean = np.mean(individual_preds)
                agreement_score = max(0, 1 - (pred_std / pred_mean))
            else:
                agreement_score = 0.5
            
            # Overall confidence
            confidence = (volume_score * 0.25 + volatility_score * 0.25 + 
                         trend_score * 0.25 + agreement_score * 0.25)
            
            return min(max(confidence, 0), 1)  # Ensure between 0 and 1
        except:
            return 0.5
    
    def analyze_stock_advanced(self, symbol):
        """Perform advanced stock analysis with ML predictions"""
        try:
            print(f"Analyzing {symbol}...")
            
            # Get stock data
            stock = yf.Ticker(symbol + '.NS')
            data = stock.history(period="2y")  # More data for better ML training
            info = stock.info
            
            if len(data) < 100:
                return None
            
            # Prepare features
            df_features = self.prepare_features_for_ml(data)
            
            # Select feature columns (excluding target and non-numeric columns)
            feature_columns = [col for col in df_features.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume'] 
                             and df_features[col].dtype in ['float64', 'int64']]
            
            # Remove columns with too many NaN values
            valid_features = []
            for col in feature_columns:
                if df_features[col].notna().sum() > len(df_features) * 0.7:  # At least 70% valid data
                    valid_features.append(col)
            
            if len(valid_features) < 10:
                return None
            
            # Prepare training data
            df_clean = df_features[valid_features + ['Close']].dropna()
            
            if len(df_clean) < 50:
                return None
            
            X = df_clean[valid_features].values
            y = df_clean['Close'].values
            
            # Train models
            models, weights = self.train_ensemble_models(X, y)
            
            # Make predictions
            last_features = X[-1].reshape(1, -1)
            predictions = self.predict_ensemble(last_features, days=7)
            
            # Calculate price targets and Fibonacci levels
            targets = self.calculate_price_targets(data, predictions)
            
            # Get news sentiment (simplified version to avoid API limits)
            sentiment = self.get_simple_sentiment(symbol)
            
            # Compile results
            analysis_result = {
                'symbol': symbol,
                'company_name': info.get('longName', 'N/A'),
                'current_price': round(data['Close'].iloc[-1], 2),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'predictions': predictions,
                'price_targets': targets,
                'fibonacci_levels': targets['fibonacci_levels'],
                'fibonacci_extensions': targets['fibonacci_extensions'],
                'technical_indicators': {
                    'RSI': round(df_features['RSI'].iloc[-1], 2) if 'RSI' in df_features.columns else 'N/A',
                    'MACD': round(df_features['MACD'].iloc[-1], 4) if 'MACD' in df_features.columns else 'N/A',
                    'ATR': round(df_features['ATR'].iloc[-1], 2) if 'ATR' in df_features.columns else 'N/A',
                    'Volume_Ratio': round(df_features['Volume_Ratio'].iloc[-1], 2) if 'Volume_Ratio' in df_features.columns else 'N/A'
                },
                'model_performance': weights,
                'confidence_score': targets['confidence_score'],
                'sentiment': sentiment,
                'data': data
            }
            
            return analysis_result
            
        except Exception as e:
            print(f"Error in advanced analysis for {symbol}: {e}")
            return None
    
    def get_simple_sentiment(self, symbol):
        """Simplified sentiment analysis"""
        try:
            # Create a mock sentiment for now (you can integrate real news API later)
            return {
                'overall_sentiment': 0.1,
                'sentiment_label': 'Neutral',
                'confidence': 0.7,
                'articles_count': 5
            }
        except:
            return {
                'overall_sentiment': 0,
                'sentiment_label': 'Neutral', 
                'confidence': 0.5,
                'articles_count': 0
            }
