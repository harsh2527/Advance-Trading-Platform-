import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class ChartVisualizer:
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
    
    def create_probability_chart(self, probabilities):
        """Create a probability pie chart"""
        labels = ['Buy', 'Sell', 'Hold']
        values = [probabilities['buy'], probabilities['sell'], probabilities['hold']]
        colors = ['#00ff00', '#ff0000', '#ffff00']
        
        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                textfont=dict(size=14, color='white')
            )
        ])
        
        fig.update_layout(
            title="Trading Probability Analysis",
            template='plotly_dark',
            height=400,
            showlegend=True
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
        data = []
        
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
            mode = "gauge+number+delta",
            value = {'HIGH': 90, 'MEDIUM': 60, 'LOW': 30}[signals['confidence']],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Action: {signals['action']} {icon}"},
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
