import yfinance as yf
import pandas as pd
import numpy as np
import ta
from tradingview_ta import TA_Handler, Interval
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from googlesearch import search
import warnings
warnings.filterwarnings('ignore')

class EnhancedStockAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def get_tradingview_analysis(self, symbol, exchange="NSE"):
        """Get TradingView technical analysis"""
        try:
            handler = TA_Handler(
                symbol=symbol,
                screener="india",
                exchange=exchange,
                interval=Interval.INTERVAL_1_DAY
            )
            analysis = handler.get_analysis()
            return analysis
        except Exception as e:
            print(f"TradingView analysis error: {e}")
            return None
    
    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance"""
        try:
            if not symbol.endswith('.NS'):
                symbol += '.NS'
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            info = stock.info
            return data, info
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None, None
    
    def get_news_sentiment(self, symbol):
        """Fetch and analyze news sentiment for the stock"""
        try:
            company_name = symbol.replace('.NS', '')
            
            # Search for recent news
            news_articles = []
            search_query = f"{company_name} stock news India"
            
            # Get search results
            search_results = list(search(search_query, num_results=10, stop=10, pause=2))
            
            sentiments = []
            articles_data = []
            
            for url in search_results[:5]:  # Limit to first 5 results
                try:
                    response = requests.get(url, timeout=10, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract title and some content
                    title = soup.find('title')
                    title_text = title.get_text() if title else ""
                    
                    # Find paragraphs for content analysis
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text() for p in paragraphs[:5]])
                    
                    if title_text and len(content) > 100:
                        # Analyze sentiment
                        combined_text = title_text + " " + content
                        blob = TextBlob(combined_text)
                        sentiment_score = blob.sentiment.polarity
                        
                        sentiments.append(sentiment_score)
                        articles_data.append({
                            'title': title_text,
                            'url': url,
                            'sentiment': sentiment_score,
                            'sentiment_label': 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'
                        })
                
                except Exception as e:
                    continue
            
            # Calculate overall sentiment
            if sentiments:
                overall_sentiment = np.mean(sentiments)
                sentiment_impact = {
                    'overall_sentiment': overall_sentiment,
                    'sentiment_label': 'Positive' if overall_sentiment > 0.1 else 'Negative' if overall_sentiment < -0.1 else 'Neutral',
                    'articles': articles_data[:3],  # Return top 3 articles
                    'confidence': min(abs(overall_sentiment) * 2, 1.0)  # Confidence based on sentiment strength
                }
            else:
                sentiment_impact = {
                    'overall_sentiment': 0,
                    'sentiment_label': 'Neutral',
                    'articles': [],
                    'confidence': 0
                }
            
            return sentiment_impact
            
        except Exception as e:
            print(f"Error in news sentiment analysis: {e}")
            return {
                'overall_sentiment': 0,
                'sentiment_label': 'Neutral',
                'articles': [],
                'confidence': 0
            }
    
    def calculate_technical_indicators(self, data):
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Price indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Low'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        
        # Momentum indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        df['MACD_Hist'] = ta.trend.macd(df['Close'])
        
        # Volume indicators
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility indicators
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Price change features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_Ratio'] = (df['Close'] - df['Open']) / df['Open']
        
        # Support and Resistance levels
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        return df
    
    def prepare_prediction_features(self, data):
        """Prepare features for machine learning models"""
        df = data.copy()
        
        # Create lag features
        for lag in [1, 2, 3, 5, 7]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'RSI_lag_{lag}'] = df['RSI'].shift(lag)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'Close_rolling_mean_{window}'] = df['Close'].rolling(window).mean()
            df[f'Close_rolling_std_{window}'] = df['Close'].rolling(window).std()
            df[f'Volume_rolling_mean_{window}'] = df['Volume'].rolling(window).mean()
        
        # Technical indicator trends
        df['RSI_trend'] = df['RSI'].diff()
        df['MACD_trend'] = df['MACD'].diff()
        df['Volume_trend'] = df['Volume_Ratio'].diff()
        
        return df
    
    def predict_price_ml(self, data, days=7):
        """Predict stock prices using multiple ML models"""
        try:
            # Prepare features
            df = self.prepare_prediction_features(data)
            
            # Define feature columns (excluding NaN-prone features)
            feature_columns = [
                'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'Volume_Ratio',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'ATR',
                'High_Low_Ratio', 'Open_Close_Ratio'
            ]
            
            # Add lag features that exist
            for col in df.columns:
                if 'lag_' in col or 'rolling_mean_' in col:
                    if col not in feature_columns:
                        feature_columns.append(col)
            
            # Remove rows with NaN values
            df_clean = df[feature_columns + ['Close']].dropna()
            
            if len(df_clean) < 50:  # Not enough data
                return None, "Insufficient data for prediction"
            
            # Prepare training data
            X = df_clean[feature_columns].values
            y = df_clean['Close'].values
            
            # Use last 80% for training, 20% for validation
            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Train multiple models
            models = {}
            
            # Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            models['Random Forest'] = rf_model
            
            # XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
            
            # LightGBM
            lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            models['LightGBM'] = lgb_model
            
            # Make predictions for next 7 days
            predictions = {}
            last_features = X[-1].reshape(1, -1)
            
            for name, model in models.items():
                model_predictions = []
                current_features = last_features.copy()
                
                for day in range(days):
                    pred_price = model.predict(current_features)[0]
                    model_predictions.append(pred_price)
                    
                    # Update features for next prediction (simple approach)
                    # In practice, you'd want more sophisticated feature updating
                    if day < days - 1:
                        new_features = current_features.copy()
                        new_features[0, 0] = pred_price  # Update close price
                        current_features = new_features
                
                predictions[name] = model_predictions
            
            # Ensemble prediction (average of all models)
            ensemble_predictions = []
            for day in range(days):
                day_predictions = [predictions[model][day] for model in predictions]
                ensemble_predictions.append(np.mean(day_predictions))
            
            # Calculate prediction confidence based on model agreement
            confidence_scores = []
            for day in range(days):
                day_predictions = [predictions[model][day] for model in predictions]
                std_dev = np.std(day_predictions)
                mean_pred = np.mean(day_predictions)
                confidence = max(0, 1 - (std_dev / mean_pred))  # Lower std = higher confidence
                confidence_scores.append(confidence)
            
            return {
                'ensemble_predictions': ensemble_predictions,
                'individual_predictions': predictions,
                'confidence_scores': confidence_scores,
                'average_confidence': np.mean(confidence_scores)
            }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def get_entry_exit_signals(self, data, predictions, sentiment):
        """Generate entry, hold, and exit signals with specific levels"""
        latest = data.iloc[-1]
        current_price = latest['Close']
        
        signals = {
            'action': 'HOLD',
            'confidence': 'MEDIUM',
            'entry_price': None,
            'stop_loss': None,
            'target_price': None,
            'risk_reward_ratio': None,
            'reasoning': []
        }
        
        # Technical analysis signals
        tech_signals = 0
        
        if latest['RSI'] < 30:
            tech_signals += 2  # Strong buy signal
            signals['reasoning'].append("RSI indicates oversold condition")
        elif latest['RSI'] > 70:
            tech_signals -= 2  # Strong sell signal
            signals['reasoning'].append("RSI indicates overbought condition")
        
        if latest['MACD'] > latest['MACD_Signal']:
            tech_signals += 1
            signals['reasoning'].append("MACD bullish crossover")
        else:
            tech_signals -= 1
        
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            tech_signals += 1
            signals['reasoning'].append("Price above both moving averages (bullish)")
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            tech_signals -= 1
            signals['reasoning'].append("Price below both moving averages (bearish)")
        
        if latest['Volume_Ratio'] > 1.5:
            tech_signals += 1
            signals['reasoning'].append("Above average volume confirmation")
        
        # Sentiment analysis influence
        sentiment_impact = sentiment['overall_sentiment']
        if sentiment['sentiment_label'] == 'Positive':
            tech_signals += 1
            signals['reasoning'].append(f"Positive news sentiment detected")
        elif sentiment['sentiment_label'] == 'Negative':
            tech_signals -= 1
            signals['reasoning'].append(f"Negative news sentiment detected")
        
        # Price prediction influence
        if predictions and len(predictions['ensemble_predictions']) >= 3:
            predicted_change = (predictions['ensemble_predictions'][2] - current_price) / current_price
            if predicted_change > 0.02:  # 2% increase predicted
                tech_signals += 1
                signals['reasoning'].append("ML models predict price increase")
            elif predicted_change < -0.02:  # 2% decrease predicted
                tech_signals -= 1
                signals['reasoning'].append("ML models predict price decrease")
        
        # Determine action based on signal strength
        if tech_signals >= 3:
            signals['action'] = 'BUY'
            signals['entry_price'] = current_price
            signals['stop_loss'] = latest['Support'] * 0.98
            signals['target_price'] = latest['Resistance'] * 1.02
            signals['confidence'] = 'HIGH' if tech_signals >= 4 else 'MEDIUM'
        elif tech_signals <= -3:
            signals['action'] = 'SELL'
            signals['entry_price'] = current_price
            signals['stop_loss'] = latest['Resistance'] * 1.02
            signals['target_price'] = latest['Support'] * 0.98
            signals['confidence'] = 'HIGH' if tech_signals <= -4 else 'MEDIUM'
        
        # Calculate risk-reward ratio
        if signals['stop_loss'] and signals['target_price']:
            risk = abs(current_price - signals['stop_loss'])
            reward = abs(signals['target_price'] - current_price)
            if risk > 0:
                signals['risk_reward_ratio'] = round(reward / risk, 2)
        
        return signals
    
    def analyze_stock(self, symbol):
        """Complete stock analysis with predictions"""
        print(f"Analyzing {symbol}...")
        
        # Get data
        data, info = self.get_stock_data(symbol)
        if data is None:
            return None
        
        # Get TradingView analysis
        clean_symbol = symbol.replace('.NS', '')
        tv_analysis = self.get_tradingview_analysis(clean_symbol)
        
        # Calculate technical indicators
        data_with_indicators = self.calculate_technical_indicators(data)
        
        # Get news sentiment
        print("Analyzing news sentiment...")
        sentiment = self.get_news_sentiment(symbol)
        
        # Predict prices
        print("Generating price predictions...")
        predictions, pred_error = self.predict_price_ml(data_with_indicators)
        
        # Get entry/exit signals
        signals = self.get_entry_exit_signals(data_with_indicators, predictions, sentiment)
        
        # Generate future dates for predictions
        future_dates = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=7,
            freq='D'
        )
        
        # Compile analysis results
        analysis = {
            'symbol': symbol,
            'current_price': round(data_with_indicators['Close'].iloc[-1], 2),
            'company_name': info.get('longName', 'N/A') if info else 'N/A',
            'market_cap': info.get('marketCap', 'N/A') if info else 'N/A',
            'pe_ratio': info.get('trailingPE', 'N/A') if info else 'N/A',
            'predictions': predictions,
            'prediction_error': pred_error,
            'future_dates': future_dates,
            'sentiment': sentiment,
            'signals': signals,
            'technical_indicators': {
                'RSI': round(data_with_indicators['RSI'].iloc[-1], 2),
                'MACD': round(data_with_indicators['MACD'].iloc[-1], 4),
                'SMA_20': round(data_with_indicators['SMA_20'].iloc[-1], 2),
                'SMA_50': round(data_with_indicators['SMA_50'].iloc[-1], 2),
                'Support': round(data_with_indicators['Support'].iloc[-1], 2),
                'Resistance': round(data_with_indicators['Resistance'].iloc[-1], 2),
                'Volume_Ratio': round(data_with_indicators['Volume_Ratio'].iloc[-1], 2)
            },
            'tradingview_summary': tv_analysis.summary if tv_analysis else None,
            'data': data_with_indicators
        }
        
        return analysis
