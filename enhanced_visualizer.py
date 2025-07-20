import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class EnhancedChartVisualizer:
    def __init__(self):
        self.colors = {
            'bullish': '#00ff00',
            'bearish': '#ff0000',
            'neutral': '#ffff00',
            'background': '#1e1e1e',
            'text': '#ffffff'
        }
    
    def create_candlestick_chart(self, data, title="Stock Price Chart"):
        """Create an interactive candlestick chart with technical indicators"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Indicators', 'Volume', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing_line_color='#00ff00',
                decreasing_line_color='#ff0000'
            ),
            row=1, col=1
        )
        
        # Moving Averages
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='purple', width=2)
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_High'],
                mode='lines',
                name='BB High',
                line=dict(color='gray', width=1),
                opacity=0.5
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['BB_Low'],
                mode='lines',
                name='BB Low',
                line=dict(color='gray', width=1),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                opacity=0.5
            ),
            row=1, col=1
        )
        
        # Support and Resistance
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Support'],
                mode='lines',
                name='Support',
                line=dict(color='green', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Resistance'],
                mode='lines',
                name='Resistance',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Volume
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
            row=2, col=1
        )
        
        # Volume SMA
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Volume_SMA'],
                mode='lines',
                name='Volume SMA',
                line=dict(color='white', width=2)
            ),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=2)
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought (70)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Oversold (30)", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", 
                     annotation_text="Neutral (50)", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            template='plotly_dark',
            showlegend=True,
            height=800,
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axis ranges
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        
        return fig
    
    def create_prediction_chart(self, historical_data, predictions, future_dates, current_price):
        """Create price prediction chart"""
        if not predictions or not predictions['ensemble_predictions']:
            return None
            
        # Get last 30 days of historical data for context
        recent_data = historical_data.tail(30)
        
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data['Close'],
                mode='lines',
                name='Historical Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Ensemble prediction
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predictions['ensemble_predictions'],
                mode='lines+markers',
                name='Ensemble Prediction',
                line=dict(color='orange', width=3, dash='dash'),
                marker=dict(size=8)
            )
        )
        
        # Individual model predictions
        colors = ['red', 'green', 'purple']
        for i, (model_name, model_predictions) in enumerate(predictions['individual_predictions'].items()):
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=model_predictions,
                    mode='lines',
                    name=f'{model_name}',
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.6
                )
            )
        
        # Current price line
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Current Price: Rs {current_price:.2f}"
        )
        
        # Calculate prediction range
        min_pred = min(predictions['ensemble_predictions'])
        max_pred = max(predictions['ensemble_predictions'])
        
        # Add confidence interval
        upper_bound = [p * 1.05 for p in predictions['ensemble_predictions']]  # +5% confidence
        lower_bound = [p * 0.95 for p in predictions['ensemble_predictions']]  # -5% confidence
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=upper_bound,
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='Confidence Interval',
                fillcolor='rgba(255, 165, 0, 0.2)'
            )
        )
        
        fig.update_layout(
            title="7-Day Price Prediction",
            template='plotly_dark',
            xaxis_title="Date",
            yaxis_title="Price (Rs)",
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_sentiment_chart(self, sentiment_data):
        """Create news sentiment visualization"""
        if not sentiment_data or not sentiment_data['articles']:
            return None
        
        fig = go.Figure()
        
        # Overall sentiment gauge
        sentiment_score = sentiment_data['overall_sentiment']
        
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"News Sentiment: {sentiment_data['sentiment_label']}"},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkgreen" if sentiment_score > 0.1 else "darkred" if sentiment_score < -0.1 else "gray"},
                'steps': [
                    {'range': [-1, -0.1], 'color': "lightcoral"},
                    {'range': [-0.1, 0.1], 'color': "lightgray"},
                    {'range': [0.1, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            title="News Sentiment Analysis"
        )
        
        return fig
    
    def create_macd_chart(self, data):
        """Create MACD chart"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('MACD Line & Signal', 'MACD Histogram'),
            row_heights=[0.7, 0.3]
        )
        
        # MACD Line and Signal
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in data['MACD_Hist']]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['MACD_Hist'],
                name='MACD Hist',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="MACD Analysis",
            template='plotly_dark',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_signals_chart(self, signals, current_price):
        """Create entry/exit signals visualization"""
        
        if signals['action'] == 'BUY':
            color = 'green'
            icon = 'BUY'
        elif signals['action'] == 'SELL':
            color = 'red'
            icon = 'SELL'
        else:
            color = 'yellow'
            icon = 'HOLD'
        
        # Create a simple gauge chart for confidence
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = {'HIGH': 90, 'MEDIUM': 60, 'LOW': 30}[signals['confidence']],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Action: {signals['action']} ({icon})"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            title=f"Trading Signal - Confidence: {signals['confidence']}",
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    def create_prediction_summary_chart(self, predictions, current_price):
        """Create a summary chart showing prediction accuracy and trends"""
        if not predictions or not predictions['ensemble_predictions']:
            return None
        
        days = list(range(1, len(predictions['ensemble_predictions']) + 1))
        ensemble_pred = predictions['ensemble_predictions']
        
        # Calculate percentage changes
        pct_changes = [(pred - current_price) / current_price * 100 for pred in ensemble_pred]
        
        fig = go.Figure()
        
        # Bar chart showing daily percentage changes
        colors = ['green' if change > 0 else 'red' for change in pct_changes]
        fig.add_trace(
            go.Bar(
                x=[f"Day {d}" for d in days],
                y=pct_changes,
                name='Predicted Change %',
                marker_color=colors,
                opacity=0.7
            )
        )
        
        # Add confidence scores as a line
        if 'confidence_scores' in predictions:
            fig.add_trace(
                go.Scatter(
                    x=[f"Day {d}" for d in days],
                    y=[c * 100 for c in predictions['confidence_scores']],  # Scale to percentage
                    mode='lines+markers',
                    name='Confidence %',
                    yaxis='y2',
                    line=dict(color='orange', width=2),
                    marker=dict(size=8)
                )
            )
        
        fig.update_layout(
            title="7-Day Prediction Summary",
            template='plotly_dark',
            xaxis_title="Day",
            yaxis_title="Predicted Change (%)",
            yaxis2=dict(
                title="Confidence (%)",
                overlaying='y',
                side='right'
            ),
            height=400,
            showlegend=True
        )
        
        return fig
