import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdvancedTradingVisualizer:
    def __init__(self):
        self.colors = {
            'bullish': '#00FF00',
            'bearish': '#FF0000',
            'neutral': '#FFFF00',
            'support': '#00BFFF',
            'resistance': '#FF4500',
            'fibonacci': "#5C29C3"
        }
    
    def create_comprehensive_chart(self, data, analysis_result, title="Advanced Stock Analysis"):
        """Create a comprehensive trading chart with all indicators"""
        
        # Create subplots
        fig = make_subplots(
            rows=4, cols=2,
            shared_xaxes=True,
            vertical_spacing=0.02,
            horizontal_spacing=0.05,
            subplot_titles=(
                'Price Action with Fibonacci & S/R', 'Volume Analysis',
                'RSI & Momentum', 'MACD',
                'Bollinger Bands & Volatility', 'Support/Resistance Zones',
                '7-Day Price Predictions', 'Model Performance'
            ),
            row_heights=[0.4, 0.2, 0.2, 0.2],
            column_widths=[0.7, 0.3],
            specs=[
                [{"colspan": 1}, {"rowspan": 2}],
                [{"colspan": 1}, None],
                [{"colspan": 1}, {"rowspan": 2}],
                [{"colspan": 1}, None]
            ]
        )
        
        # Main candlestick chart with Fibonacci levels
        self.add_main_candlestick_chart(fig, data, analysis_result, row=1, col=1)
        
        # Volume analysis
        self.add_volume_chart(fig, data, row=2, col=1)
        
        # RSI and momentum indicators
        self.add_rsi_momentum_chart(fig, data, analysis_result, row=3, col=1)
        
        # MACD
        self.add_macd_chart(fig, data, analysis_result, row=4, col=1)
        
        # Volume analysis detailed
        self.add_volume_analysis_chart(fig, data, row=1, col=2)
        
        # Support/Resistance zones
        self.add_support_resistance_chart(fig, data, analysis_result, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#FFFFFF'}
            },
            template='plotly_dark',
            height=1200,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def add_main_candlestick_chart(self, fig, data, analysis_result, row, col):
        """Add main candlestick chart with Fibonacci levels"""
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=row, col=col
        )
        
        # Moving averages
        if len(data) >= 20:
            sma_20 = data['Close'].rolling(window=20).mean()
            sma_50 = data['Close'].rolling(window=50).mean()
            
            fig.add_trace(
                go.Scatter(x=data.index, y=sma_20, mode='lines',
                          name='SMA 20', line=dict(color='orange', width=1)),
                row=row, col=col
            )
            
            if len(data) >= 50:
                fig.add_trace(
                    go.Scatter(x=data.index, y=sma_50, mode='lines',
                              name='SMA 50', line=dict(color='purple', width=1)),
                    row=row, col=col
                )
        
        # Fibonacci levels
        if 'fibonacci_levels' in analysis_result:
            fib_levels = analysis_result['fibonacci_levels']
            for level_name, level_price in fib_levels.items():
                fig.add_hline(
                    y=level_price,
                    line_dash="dash",
                    line_color=self.colors['fibonacci'],
                    annotation_text=f"Fib {level_name}: ₹{level_price:.2f}",
                    annotation_position="top right",
                    row=row, col=col
                )
        
        # Support and Resistance
        if 'price_targets' in analysis_result:
            tech_levels = analysis_result['price_targets']['technical_levels']
            
            fig.add_hline(
                y=tech_levels['resistance'],
                line_dash="dot",
                line_color=self.colors['resistance'],
                annotation_text=f"Resistance: ₹{tech_levels['resistance']:.2f}",
                row=row, col=col
            )
            
            fig.add_hline(
                y=tech_levels['support'],
                line_dash="dot", 
                line_color=self.colors['support'],
                annotation_text=f"Support: ₹{tech_levels['support']:.2f}",
                row=row, col=col
            )
        
        # Current price line
        current_price = analysis_result['current_price']
        fig.add_hline(
            y=current_price,
            line_color='white',
            line_width=2,
            annotation_text=f"Current: ₹{current_price:.2f}",
            annotation_position="bottom right",
            row=row, col=col
        )
    
    def add_volume_chart(self, fig, data, row, col):
        """Add volume analysis chart"""
        
        # Volume bars with color coding
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=row, col=col
        )
        
        # Volume moving average
        if len(data) >= 20:
            volume_ma = data['Volume'].rolling(window=20).mean()
            fig.add_trace(
                go.Scatter(x=data.index, y=volume_ma, mode='lines',
                          name='Volume MA', line=dict(color='white', width=2)),
                row=row, col=col
            )
    
    def add_rsi_momentum_chart(self, fig, data, analysis_result, row, col):
        """Add RSI and momentum indicators"""
        
        # Calculate RSI if not in analysis
        rsi = data['Close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() / \
              data['Close'].diff().abs().rolling(window=14).mean() * 100
        
        fig.add_trace(
            go.Scatter(x=data.index, y=rsi, mode='lines',
                      name='RSI', line=dict(color='blue', width=2)),
            row=row, col=col
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red",
                     annotation_text="Overbought (70)", row=row, col=col)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Oversold (30)", row=row, col=col)
        fig.add_hline(y=50, line_dash="dot", line_color="gray",
                     annotation_text="Neutral (50)", row=row, col=col)
    
    def add_macd_chart(self, fig, data, analysis_result, row, col):
        """Add MACD chart"""
        
        # Calculate MACD
        ema_12 = data['Close'].ewm(span=12).mean()
        ema_26 = data['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        fig.add_trace(
            go.Scatter(x=data.index, y=macd, mode='lines',
                      name='MACD', line=dict(color='blue', width=2)),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(x=data.index, y=signal, mode='lines',
                      name='Signal', line=dict(color='red', width=2)),
            row=row, col=col
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in histogram]
        fig.add_trace(
            go.Bar(x=data.index, y=histogram, name='MACD Hist',
                  marker_color=colors, opacity=0.7),
            row=row, col=col
        )
    
    def add_volume_analysis_chart(self, fig, data, row, col):
        """Add detailed volume analysis"""
        
        # Volume profile (simplified)
        recent_data = data.tail(50)
        volume_by_price = recent_data.groupby(pd.cut(recent_data['Close'], bins=20))['Volume'].sum()
        
        # Create horizontal bar chart for volume profile
        price_levels = [(interval.left + interval.right) / 2 for interval in volume_by_price.index]
        volumes = volume_by_price.values
        
        fig.add_trace(
            go.Bar(x=volumes, y=price_levels, orientation='h',
                  name='Volume Profile', marker_color='cyan', opacity=0.6),
            row=row, col=col
        )
    
    def add_support_resistance_chart(self, fig, data, analysis_result, row, col):
        """Add support/resistance zone analysis"""
        
        # Price distribution
        recent_prices = data['Close'].tail(100)
        
        fig.add_trace(
            go.Histogram(x=recent_prices, nbinsx=20, name='Price Distribution',
                        marker_color='yellow', opacity=0.7),
            row=row, col=col
        )
        
        # Mark current price
        current_price = analysis_result['current_price']
        fig.add_vline(x=current_price, line_dash="dash", line_color="white",
                     annotation_text=f"Current: ₹{current_price:.2f}",
                     row=row, col=col)
    
    def create_prediction_chart(self, analysis_result):
        """Create 7-day price prediction chart"""
        
        if 'predictions' not in analysis_result or not analysis_result['predictions']:
            return None
        
        predictions = analysis_result['predictions']['ensemble_prediction']
        dates = pd.date_range(start=datetime.now(), periods=7, freq='D')
        
        fig = go.Figure()
        
        # Prediction line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='orange', width=3),
                marker=dict(size=8, color='orange')
            )
        )
        
        # Add confidence bands
        current_price = analysis_result['current_price']
        confidence = analysis_result.get('confidence_score', 0.5)
        
        upper_band = [p * (1 + 0.1 * (1 - confidence)) for p in predictions]
        lower_band = [p * (1 - 0.1 * (1 - confidence)) for p in predictions]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=upper_band,
                fill=None,
                mode='lines',
                line=dict(width=0),
                name='Upper Confidence'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=lower_band,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='Confidence Band',
                fillcolor='rgba(255, 165, 0, 0.2)'
            )
        )
        
        # Current price reference
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Current: ₹{current_price:.2f}"
        )
        
        fig.update_layout(
            title="7-Day Price Prediction",
            template='plotly_dark',
            height=400,
            xaxis_title="Date",
            yaxis_title="Price (₹)"
        )
        
        return fig
    
    def create_fibonacci_levels_chart(self, data, analysis_result):
        """Create detailed Fibonacci levels chart"""
        
        fig = go.Figure()
        
        # Recent price data
        recent_data = data.tail(50)
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=recent_data.index,
                open=recent_data['Open'],
                high=recent_data['High'],
                low=recent_data['Low'],
                close=recent_data['Close'],
                name="Price"
            )
        )
        
        # Fibonacci retracement levels
        if 'fibonacci_levels' in analysis_result:
            fib_levels = analysis_result['fibonacci_levels']
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
            
            for i, (level_name, level_price) in enumerate(fib_levels.items()):
                fig.add_hline(
                    y=level_price,
                    line_dash="dash",
                    line_color=colors[i % len(colors)],
                    annotation_text=f"{level_name}: ₹{level_price:.2f}",
                    annotation_position="right"
                )
        
        # Fibonacci extensions
        if 'fibonacci_extensions' in analysis_result:
            fib_ext = analysis_result['fibonacci_extensions']
            
            for level_name, level_price in fib_ext.items():
                fig.add_hline(
                    y=level_price,
                    line_dash="dot",
                    line_color="purple",
                    annotation_text=f"Ext {level_name}: ₹{level_price:.2f}",
                    annotation_position="left"
                )
        
        fig.update_layout(
            title="Fibonacci Retracement & Extension Levels",
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    def create_trading_dashboard_metrics(self, analysis_result):
        """Create trading metrics dashboard"""
        
        current_price = analysis_result['current_price']
        predictions = analysis_result.get('predictions', {})
        
        if predictions and 'ensemble_prediction' in predictions:
            pred_prices = predictions['ensemble_prediction']
            
            # Calculate expected metrics
            day1_price = pred_prices[0] if len(pred_prices) > 0 else current_price
            day7_price = pred_prices[-1] if len(pred_prices) > 6 else current_price
            
            expected_return_1d = ((day1_price - current_price) / current_price) * 100
            expected_return_7d = ((day7_price - current_price) / current_price) * 100
            
            # High and low predictions
            pred_high = max(pred_prices)
            pred_low = min(pred_prices)
            
            # Support and resistance
            tech_levels = analysis_result.get('price_targets', {}).get('technical_levels', {})
            support = tech_levels.get('support', current_price * 0.95)
            resistance = tech_levels.get('resistance', current_price * 1.05)
            
            # Risk/Reward calculations
            upside_potential = ((resistance - current_price) / current_price) * 100
            downside_risk = ((current_price - support) / current_price) * 100
            risk_reward_ratio = upside_potential / downside_risk if downside_risk > 0 else 0
            
            metrics = {
                'Current Price': f"₹{current_price:.2f}",
                'Expected Day 1': f"₹{day1_price:.2f} ({expected_return_1d:+.2f}%)",
                'Expected Day 7': f"₹{day7_price:.2f} ({expected_return_7d:+.2f}%)",
                'Predicted High': f"₹{pred_high:.2f} ({((pred_high-current_price)/current_price)*100:+.2f}%)",
                'Predicted Low': f"₹{pred_low:.2f} ({((pred_low-current_price)/current_price)*100:+.2f}%)",
                'Next Resistance': f"₹{resistance:.2f} ({upside_potential:+.2f}%)",
                'Next Support': f"₹{support:.2f} ({-downside_risk:.2f}%)",
                'Risk/Reward Ratio': f"{risk_reward_ratio:.2f}",
                'Confidence Score': f"{analysis_result.get('confidence_score', 0.5):.1%}",
                'Volatility': f"{((pred_high-pred_low)/current_price)*100:.2f}%"
            }
        else:
            metrics = {
                'Current Price': f"₹{current_price:.2f}",
                'Status': 'Prediction data not available'
            }
        
        return metrics
    
    def create_model_performance_chart(self, analysis_result):
        """Create model performance comparison chart"""
        
        if 'model_performance' not in analysis_result:
            return None
        
        model_weights = analysis_result['model_performance']
        
        fig = go.Figure()
        
        # Bar chart of model weights
        fig.add_trace(
            go.Bar(
                x=list(model_weights.keys()),
                y=list(model_weights.values()),
                name='Model Weights',
                marker_color=['red', 'green', 'blue', 'orange', 'purple'][:len(model_weights)]
            )
        )
        
        fig.update_layout(
            title="ML Model Performance Weights",
            template='plotly_dark',
            height=300,
            xaxis_title="Models",
            yaxis_title="Weight"
        )
        
        return fig
