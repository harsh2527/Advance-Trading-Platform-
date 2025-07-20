import yfinance as yf
import threading
import time
from datetime import datetime
import pandas as pd
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
import numpy as np
from tradingview_ta import TA_Handler, Interval
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import warnings
warnings.filterwarnings('ignore')

class EnhancedStockAnalyzer:
    def __init__(self):
        pass
    
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
            return analysis.summary
        except Exception as e:
            print(f"TradingView analysis error: {e}")
            return None

    def get_news_sentiment(self, symbol):
        """Fetch and analyze news sentiment for the stock"""
        try:
            company_name = symbol.replace('.NS', '')
            
            # Search for recent news
            search_query = f"{company_name} stock news India"
            
            # Get search results
            search_results = list(search(search_query, num_results=5, pause=2))
            
            sentiments = []
            articles_data = []
            
            for url in search_results[:5]:  # Limit to first 5 results
                try:
                    response = requests.get(url, timeout=10)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    title = soup.find('title')
                    title_text = title.get_text() if title else ""
                    paragraphs = soup.find_all('p')
                    content = ' '.join([p.get_text() for p in paragraphs[:2]])
                    if title_text and len(content) > 100:
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
                except Exception:
                    continue
            
            overall_sentiment = np.mean(sentiments) if sentiments else 0
            sentiment_impact = {
                'overall_sentiment': overall_sentiment,
                'sentiment_label': 'Positive' if overall_sentiment > 0.1 else 'Negative' if overall_sentiment < -0.1 else 'Neutral',
                'articles': articles_data
            }
            
            return sentiment_impact
        except Exception as e:
            print(f"Error in news sentiment analysis: {e}")
            return {
                'overall_sentiment': 0,
                'sentiment_label': 'Neutral',
                'articles': []
            }

    def predict_price_ml(self, data, days=7):
        """Predict stock prices using ML model"""
        try:
            df = data.copy()
            # Add Volume_Ratio and ATR calculations
            df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20)
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
            
            feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'Volume_Ratio', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'ATR']
            
            # Only use columns that exist and have no NaN
            available_columns = [col for col in feature_columns if col in df.columns]
            df_clean = df[available_columns].dropna()
            
            if len(df_clean) < 30:  # Need minimum data
                # Simple prediction based on trend
                recent_prices = data['Close'].tail(7).values
                trend = np.mean(np.diff(recent_prices))
                current_price = data['Close'].iloc[-1]
                predicted_values = [current_price + (i+1)*trend for i in range(days)]
                return predicted_values
            
            X = df_clean[available_columns].values[:-1]
            y = df_clean['Close'].values[1:]
            model = xgb.XGBRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Predict next 7 days
            predicted_values = []
            last_features = df_clean[available_columns].iloc[-1].values.copy()
            
            for day in range(days):
                pred_price = model.predict(last_features.reshape(1, -1))[0]
                predicted_values.append(pred_price)
                
                # Update features for next prediction (simple approach)
                if len(last_features) > 0:
                    last_features[0] = pred_price  # Update close price
            
            return predicted_values
        except Exception as e:
            print(f"Error in ML prediction: {e}")
            # Fallback to simple trend prediction
            try:
                recent_prices = data['Close'].tail(7).values
                trend = np.mean(np.diff(recent_prices))
                current_price = data['Close'].iloc[-1]
                predicted_values = [current_price + (i+1)*trend for i in range(days)]
                return predicted_values
            except:
                return [data['Close'].iloc[-1]] * days
        
    def analyze_stock(self, symbol):
        """Complete stock analysis"""
        try:
            stock = yf.Ticker(symbol + '.NS')
            data = stock.history(period="1y")
            info = stock.info
            data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
            data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
            data['EMA_12'] = ta.trend.ema_indicator(data['Close'], window=12)
            data['EMA_26'] = ta.trend.ema_indicator(data['Close'], window=26)
            data['RSI'] = ta.momentum.rsi(data['Close'])
            data['MACD'] = ta.trend.macd(data['Close'])
            data['MACD_Signal'] = ta.trend.macd_signal(data['Close'])
            predictions = self.predict_price_ml(data)
            sentiment = self.get_news_sentiment(symbol)
            
            return {
                'symbol': symbol,
                'current_price': round(data['Close'].iloc[-1], 2),
                'company_name': info.get('longName', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'predictions': predictions,
                'sentiment': sentiment
            }
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
            return None

class EnhancedChartVisualizer:
    def create_candlestick(self, data, title):
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Price"
        ))
        fig.update_layout(title=title)
        return fig

    def create_future_prediction_chart(self, predictions, title):
        days = pd.date_range(start=datetime.now(), periods=len(predictions), freq='D')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=days, y=predictions, mode='lines+markers', name='Predicted Close'))
        fig.update_layout(title=title)
        return fig

