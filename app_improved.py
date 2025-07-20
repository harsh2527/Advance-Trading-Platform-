import streamlit as st
import pandas as pd
import numpy as np
from enhanced_modules import EnhancedStockAnalyzer, EnhancedChartVisualizer
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Improved Real-Time Stock Trader", layout="wide")

st.title("📊 Real-Time Stock Trading Platform")
st.markdown("### Advanced Portfolio management with enhanced predictions and sentiment analysis")

analyzer = EnhancedStockAnalyzer()
visualizer = EnhancedChartVisualizer()

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['RELIANCE', 'TCS', 'INFY']

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Sidebar
st.sidebar.header("Controls")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="TCS").upper()
analyze_button = st.sidebar.button("Analyze Stock")

if analyze_button:
    with st.spinner(f"Analyzing {symbol}..."):
        try:
            analysis_result = analyzer.analyze_stock(symbol)
            st.session_state.analysis_result = analysis_result
        except Exception as e:
            st.error(f"Error: {str(e)}")

if st.session_state.analysis_result:
    analysis_result = st.session_state.analysis_result
    st.header(f"Analysis for {analysis_result['symbol']}")
    st.metric("Current Price", f"₹{analysis_result['current_price']}")
    st.metric("Market Cap", f"₹{analysis_result['market_cap']:,}")
    st.metric("P/E Ratio", analysis_result['pe_ratio'])
    
    prediction_chart = visualizer.create_future_prediction_chart(analysis_result['predictions'], "7-Day Price Prediction")
    st.plotly_chart(prediction_chart, use_container_width=True)
    
    sentiment = analysis_result['sentiment']
    st.subheader("News Sentiment")
    st.write(f"Overall Sentiment: {sentiment['sentiment_label']}")
    for article in sentiment['articles'][:3]:
        st.markdown(f"- {article['title']} - {article['sentiment_label']}")

# Footer
st.markdown("---")
st.markdown("**Disclaimer:** Educational purposes only. Do your research before investing.")

