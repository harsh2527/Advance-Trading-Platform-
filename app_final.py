import streamlit as st
import pandas as pd
import numpy as np
from enhanced_analyzer import EnhancedStockAnalyzer
from enhanced_visualizer import EnhancedChartVisualizer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Enhanced Stock Trader", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: linear-gradient(90deg, #1e2130 0%, #262730 100%);
    padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #00ff00; margin: 0.5rem 0;
}
.buy-signal { border-left-color: #00ff00 !important; }
.sell-signal { border-left-color: #ff0000 !important; }
.hold-signal { border-left-color: #ffff00 !important; }
</style>
""", unsafe_allow_html=True)

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = EnhancedStockAnalyzer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = EnhancedChartVisualizer()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

st.title("Enhanced Stock Trader with AI Predictions")
st.markdown("### Advanced Stock Analysis with ML Price Predictions & News Sentiment")

# Sidebar
st.sidebar.header("Analysis Controls")
symbol = st.sidebar.text_input("Enter Stock Symbol", value="TCS").upper()
analyze_button = st.sidebar.button("Analyze Stock", type="primary")

popular_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT", "SBIN"]
selected_popular = st.sidebar.selectbox("Quick Select", [""] + popular_stocks)
if selected_popular:
    symbol = selected_popular

# Main analysis
if analyze_button or st.session_state.analysis_results is not None:
    if analyze_button:
        with st.spinner(f"Analyzing {symbol}... This may take a few minutes."):
            try:
                analysis = st.session_state.analyzer.analyze_stock(symbol)
                st.session_state.analysis_results = analysis
            except Exception as e:
                st.error(f"Error analyzing {symbol}: {str(e)}")
                st.stop()
    
    analysis = st.session_state.analysis_results
    if analysis is None:
        st.error("No data available for this symbol.")
        st.stop()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"Rs {analysis['current_price']}")
    with col2:
        st.metric("Market Cap", f"Rs {analysis['market_cap']:,.0f}" if isinstance(analysis['market_cap'], (int, float)) else "N/A")
    with col3:
        st.metric("P/E Ratio", f"{analysis['pe_ratio']:.2f}" if isinstance(analysis['pe_ratio'], (int, float)) else "N/A")
    with col4:
        rsi = analysis['technical_indicators']['RSI']
        st.metric("RSI", f"{rsi:.2f}", "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral")
    
    # Main content
    col_main, col_pred = st.columns([2, 1])
    
    with col_main:
        st.subheader("Price Chart & Technical Indicators")
        chart = st.session_state.visualizer.create_candlestick_chart(
            analysis['data'], title=f"{analysis['company_name']} ({symbol})"
        )
        st.plotly_chart(chart, use_container_width=True)
    
    with col_pred:
        st.subheader("7-Day Price Predictions")
        if analysis['predictions'] and not analysis['prediction_error']:
            predictions = analysis['predictions']
            pred_chart = st.session_state.visualizer.create_prediction_chart(
                analysis['data'], predictions, analysis['future_dates'], analysis['current_price']
            )
            if pred_chart:
                st.plotly_chart(pred_chart, use_container_width=True)
            
            # Display predictions
            pred_df = pd.DataFrame({
                'Date': analysis['future_dates'].strftime('%Y-%m-%d'),
                'Price': [f"Rs {p:.2f}" for p in predictions['ensemble_predictions']],
                'Change': [f"{((p - analysis['current_price']) / analysis['current_price'] * 100):.2f}%" 
                          for p in predictions['ensemble_predictions']]
            })
            st.dataframe(pred_df)
        else:
            st.warning("Predictions not available")
    
    # Trading signals
    st.subheader("Trading Signals & Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        signals = analysis['signals']
        signal_color = "buy-signal" if signals['action'] == 'BUY' else "sell-signal" if signals['action'] == 'SELL' else "hold-signal"
        
        st.markdown(f"""
        <div class="metric-card {signal_color}">
            <h3>Action: {signals['action']}</h3>
            <p><strong>Confidence:</strong> {signals['confidence']}</p>
            <p><strong>Entry:</strong> Rs {signals['entry_price']:.2f if signals['entry_price'] else 'N/A'}</p>
            <p><strong>Stop Loss:</strong> Rs {signals['stop_loss']:.2f if signals['stop_loss'] else 'N/A'}</p>
            <p><strong>Target:</strong> Rs {signals['target_price']:.2f if signals['target_price'] else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if signals['reasoning']:
            st.write("**Reasoning:**")
            for reason in signals['reasoning']:
                st.write(f"• {reason}")
    
    with col2:
        sentiment = analysis['sentiment']
        st.write(f"**News Sentiment:** {sentiment['sentiment_label']} ({sentiment['overall_sentiment']:.2f})")
        
        if sentiment['articles']:
            st.write("**Recent News:**")
            for article in sentiment['articles'][:2]:
                st.write(f"• {article['title'][:80]}... ({article['sentiment_label']})")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Disclaimer:</strong> Educational purposes only. Do your own research before investing.</p>
</div>
""", unsafe_allow_html=True)
