import yfinance as yf
import pandas as pd
import numpy as np
import ta
from tradingview_ta import TA_Handler, Interval
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class AdvancedStockAnalyzer:
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
        
        # Volatility indicators
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        # Support and Resistance levels
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        return df
    
    def calculate_probability_score(self, data, tv_analysis=None):
        """Calculate probability scores for buy/sell/hold decisions"""
        latest = data.iloc[-1]
        scores = {
            'buy_signals': 0,
            'sell_signals': 0,
            'neutral_signals': 0
        }
        
        # Technical Analysis Scoring
        # RSI Analysis
        if latest['RSI'] < 30:
            scores['buy_signals'] += 2  # Oversold
        elif latest['RSI'] > 70:
            scores['sell_signals'] += 2  # Overbought
        elif 40 <= latest['RSI'] <= 60:
            scores['neutral_signals'] += 1
        
        # MACD Analysis
        if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > 0:
            scores['buy_signals'] += 2
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < 0:
            scores['sell_signals'] += 2
        
        # Moving Average Analysis
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            scores['buy_signals'] += 2
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            scores['sell_signals'] += 2
        
        # Bollinger Bands Analysis
        if latest['Close'] <= latest['BB_Low']:
            scores['buy_signals'] += 1  # Price touching lower band
        elif latest['Close'] >= latest['BB_High']:
            scores['sell_signals'] += 1  # Price touching upper band
        
        # Volume Analysis
        if latest['Volume'] > latest['Volume_SMA'] * 1.5:
            scores['buy_signals'] += 1  # High volume confirmation
        
        # TradingView Analysis Integration
        if tv_analysis:
            tv_recommendation = tv_analysis.summary['RECOMMENDATION']
            if tv_recommendation == 'STRONG_BUY':
                scores['buy_signals'] += 3
            elif tv_recommendation == 'BUY':
                scores['buy_signals'] += 2
            elif tv_recommendation == 'STRONG_SELL':
                scores['sell_signals'] += 3
            elif tv_recommendation == 'SELL':
                scores['sell_signals'] += 2
            else:
                scores['neutral_signals'] += 1
        
        # Calculate probabilities
        total_signals = sum(scores.values())
        if total_signals == 0:
            return {'buy': 33.33, 'sell': 33.33, 'hold': 33.34}
        
        buy_prob = (scores['buy_signals'] / total_signals) * 100
        sell_prob = (scores['sell_signals'] / total_signals) * 100
        hold_prob = (scores['neutral_signals'] / total_signals) * 100
        
        # Normalize to ensure sum is 100%
        remaining = 100 - (buy_prob + sell_prob + hold_prob)
        hold_prob += remaining
        
        return {
            'buy': round(buy_prob, 2),
            'sell': round(sell_prob, 2),
            'hold': round(hold_prob, 2)
        }
    
    def get_entry_exit_signals(self, data, probabilities):
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
        
        # Determine primary action based on probabilities
        max_prob = max(probabilities.values())
        primary_action = [k for k, v in probabilities.items() if v == max_prob][0]
        
        if primary_action == 'buy' and probabilities['buy'] > 50:
            signals['action'] = 'BUY'
            signals['entry_price'] = current_price
            signals['stop_loss'] = latest['Support'] * 0.98  # 2% below support
            signals['target_price'] = latest['Resistance'] * 1.02  # 2% above resistance
            
            if probabilities['buy'] > 70:
                signals['confidence'] = 'HIGH'
            elif probabilities['buy'] > 60:
                signals['confidence'] = 'MEDIUM'
            else:
                signals['confidence'] = 'LOW'
                
        elif primary_action == 'sell' and probabilities['sell'] > 50:
            signals['action'] = 'SELL'
            signals['entry_price'] = current_price
            signals['stop_loss'] = latest['Resistance'] * 1.02  # 2% above resistance
            signals['target_price'] = latest['Support'] * 0.98  # 2% below support
            
            if probabilities['sell'] > 70:
                signals['confidence'] = 'HIGH'
            elif probabilities['sell'] > 60:
                signals['confidence'] = 'MEDIUM'
            else:
                signals['confidence'] = 'LOW'
        
        # Calculate risk-reward ratio
        if signals['stop_loss'] and signals['target_price']:
            risk = abs(current_price - signals['stop_loss'])
            reward = abs(signals['target_price'] - current_price)
            if risk > 0:
                signals['risk_reward_ratio'] = round(reward / risk, 2)
        
        # Add reasoning
        if latest['RSI'] < 30:
            signals['reasoning'].append("RSI indicates oversold condition")
        if latest['RSI'] > 70:
            signals['reasoning'].append("RSI indicates overbought condition")
        if latest['Close'] > latest['SMA_20']:
            signals['reasoning'].append("Price above 20-day SMA (bullish)")
        if latest['MACD'] > latest['MACD_Signal']:
            signals['reasoning'].append("MACD bullish crossover")
        if latest['Volume'] > latest['Volume_SMA']:
            signals['reasoning'].append("Above average volume confirmation")
        
        return signals
    
    def analyze_stock(self, symbol):
        """Complete stock analysis"""
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
        
        # Calculate probabilities
        probabilities = self.calculate_probability_score(data_with_indicators, tv_analysis)
        
        # Get entry/exit signals
        signals = self.get_entry_exit_signals(data_with_indicators, probabilities)
        
        # Compile analysis results
        analysis = {
            'symbol': symbol,
            'current_price': round(data_with_indicators['Close'].iloc[-1], 2),
            'company_name': info.get('longName', 'N/A') if info else 'N/A',
            'market_cap': info.get('marketCap', 'N/A') if info else 'N/A',
            'pe_ratio': info.get('trailingPE', 'N/A') if info else 'N/A',
            'probabilities': probabilities,
            'signals': signals,
            'technical_indicators': {
                'RSI': round(data_with_indicators['RSI'].iloc[-1], 2),
                'MACD': round(data_with_indicators['MACD'].iloc[-1], 4),
                'SMA_20': round(data_with_indicators['SMA_20'].iloc[-1], 2),
                'SMA_50': round(data_with_indicators['SMA_50'].iloc[-1], 2),
                'Support': round(data_with_indicators['Support'].iloc[-1], 2),
                'Resistance': round(data_with_indicators['Resistance'].iloc[-1], 2),
                'Volume_Ratio': round(data_with_indicators['Volume'].iloc[-1] / data_with_indicators['Volume_SMA'].iloc[-1], 2)
            },
            'tradingview_summary': tv_analysis.summary if tv_analysis else None,
            'data': data_with_indicators
        }
        
        return analysis
