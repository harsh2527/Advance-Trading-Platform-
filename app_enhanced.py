import streamlit as st
import pandas as pd
import numpy as np
from enhanced_analyzer import EnhancedStockAnalyzer
from enhanced_visualizer import EnhancedChartVisualizer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Enhanced Stock Trader",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        background-color: #1e2130;
    }
    .metric-card {
        background: linear-gradient(90deg, #1e2130 0%, #262730 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff00;
        margin: 0.5rem 0;
    }
    .buy-signal {
        border-left-color: #00ff00 !important;
    }
    .sell-signal {
        border-left-color: #ff0000 !important;
    }
    .hold-signal {
        border-left-color: #ffff00 !important;
    }
    .prediction-card {
        background: linear-gradient(45deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = EnhancedStockAnalyzer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = EnhancedChartVisualizer()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Header
st.title("Enhanced Stock Trader with AI Predictions")
st.markdown("### Advanced Stock Analysis with ML Price Predictions & News Sentiment")

# Sidebar
st.sidebar.header("Analysis Controls")

# Stock symbol input
symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="TCS",
    help="Enter NSE stock symbol (e.g., RELIANCE, TCS, INFY)"
).upper()

# Analysis period
period = st.sidebar.selectbox(
    "Analysis Period",
    options=["3mo", "6mo", "1y", "2y"],
    index=2,
    help="Select the time period for analysis"
)

# Analysis button
analyze_button = st.sidebar.button("Analyze Stock", type="primary")

# Popular stocks quick select
st.sidebar.subheader("Popular Stocks")
popular_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT", "SBIN", "BHARTIARTL"]
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
        st.error("No data available for this symbol. Please check the symbol and try again.")
        st.stop()
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    # Basic info metrics
    with col1:
        st.metric(
            label="Current Price",
            value=f"Rs {analysis['current_price']}",
        )
    
    with col2:
        st.metric(
            label="Market Cap",
            value=f"Rs {analysis['market_cap']:,.0f}" if isinstance(analysis['market_cap'], (int, float)) else "N/A"
        )
    
    with col3:
        st.metric(
            label="P/E Ratio",
            value=f"{analysis['pe_ratio']:.2f}" if isinstance(analysis['pe_ratio'], (int, float)) else "N/A"
        )
    
    with col4:
        rsi_value = analysis['technical_indicators']['RSI']
        rsi_status = "Oversold" if rsi_value < 30 else "Overbought" if rsi_value > 70 else "Neutral"
        st.metric(
            label="RSI",
            value=f"{rsi_value:.2f}",
            delta=rsi_status
        )
    
    # Main content area with three columns
    col_main, col_pred, col_sentiment = st.columns([2, 1.2, 0.8])
    
    with col_main:
        # Price chart
        st.subheader("Price Chart & Technical Indicators")
        chart = st.session_state.visualizer.create_candlestick_chart(
            analysis['data'], 
            title=f"{analysis['company_name']} ({symbol})"
        )
        st.plotly_chart(chart, use_container_width=True)
        
        # MACD chart
        st.subheader("MACD Analysis")
        macd_chart = st.session_state.visualizer.create_macd_chart(analysis['data'])
        st.plotly_chart(macd_chart, use_container_width=True)
    
    with col_pred:
        # Price Predictions
        st.subheader("7-Day Price Predictions")
        
        if analysis['predictions'] and not analysis['prediction_error']:
            predictions = analysis['predictions']
            
            # Prediction chart
            pred_chart = st.session_state.visualizer.create_prediction_chart(
                analysis['data'], 
                predictions, 
                analysis['future_dates'], 
                analysis['current_price']
            )
            if pred_chart:
                st.plotly_chart(pred_chart, use_container_width=True)
            
            # Prediction summary
            summary_chart = st.session_state.visualizer.create_prediction_summary_chart(
                predictions, 
                analysis['current_price']
            )
            if summary_chart:
                st.plotly_chart(summary_chart, use_container_width=True)
            
            # Display prediction values
            st.subheader("Predicted Prices")
            pred_df = pd.DataFrame({
                'Date': analysis['future_dates'].strftime('%Y-%m-%d'),
                'Predicted Price': [f"Rs {p:.2f}" for p in predictions['ensemble_predictions']],
                'Change (%)': [f"{((p - analysis['current_price']) / analysis['current_price'] * 100):.2f}%" 
                              for p in predictions['ensemble_predictions']],
                'Confidence': [f"{c:.1%}" for c in predictions['confidence_scores']] if 'confidence_scores' in predictions else ['N/A'] * 7
            })
            st.dataframe(pred_df, use_container_width=True)
            
            # Overall prediction confidence
            avg_confidence = predictions.get('average_confidence', 0)
            st.markdown(f"""
            <div class="prediction-card">
                <h4>Prediction Confidence: {avg_confidence:.1%}</h4>
                <p>Based on ensemble of Random Forest, XGBoost, and LightGBM models</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.warning("Price predictions not available")
            if analysis['prediction_error']:
                st.error(f"Error: {analysis['prediction_error']}")
    
    with col_sentiment:
        # News Sentiment Analysis
        st.subheader("News Sentiment")
        
        sentiment = analysis['sentiment']
        
        # Sentiment gauge
        sentiment_chart = st.session_state.visualizer.create_sentiment_chart(sentiment)
        if sentiment_chart:
            st.plotly_chart(sentiment_chart, use_container_width=True)
        
        # Trading signal
        st.subheader("Trading Signal")
        signal_chart = st.session_state.visualizer.create_signals_chart(
            analysis['signals'], 
            analysis['current_price']
        )
        st.plotly_chart(signal_chart, use_container_width=True)
        
        # Recent news articles
        if sentiment['articles']:
            st.subheader("Recent News")
            for i, article in enumerate(sentiment['articles']):
                sentiment_color = "green" if article['sentiment'] > 0.1 else "red" if article['sentiment'] < -0.1 else "gray"
                st.markdown(f"""
                <div style="border-left: 3px solid {sentiment_color}; padding: 10px; margin: 5px 0; background: rgba(255,255,255,0.05);">
                    <small><strong>{article['sentiment_label']}</strong> ({article['sentiment']:.2f})</small><br>
                    <span style="font-size: 0.9em;">{article['title'][:100]}...</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed Analysis Section
    st.header("Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Technical indicators
        st.subheader("Technical Indicators")
        tech_df = pd.DataFrame([analysis['technical_indicators']]).T
        tech_df.columns = ['Value']
        st.dataframe(tech_df, use_container_width=True)
        
        # TradingView summary
        if analysis['tradingview_summary']:
            st.subheader("TradingView Summary")
            tv_summary = analysis['tradingview_summary']
            st.markdown(f"**Recommendation:** {tv_summary['RECOMMENDATION']}")
            st.markdown(f"**Buy Signals:** {tv_summary['BUY']}")
            st.markdown(f"**Sell Signals:** {tv_summary['SELL']}")
            st.markdown(f"**Neutral Signals:** {tv_summary['NEUTRAL']}")
    
    with col2:
        # Trading signals
        st.subheader("Trading Signals")
        signals = analysis['signals']
        
        # Signal card
        signal_color = "buy-signal" if signals['action'] == 'BUY' else \
                      "sell-signal" if signals['action'] == 'SELL' else "hold-signal"
        
        entry_price_str = f"Rs {signals['entry_price']:.2f}" if signals['entry_price'] else 'N/A'
        stop_loss_str = f"Rs {signals['stop_loss']:.2f}" if signals['stop_loss'] else 'N/A'
        target_price_str = f"Rs {signals['target_price']:.2f}" if signals['target_price'] else 'N/A'
        
        st.markdown(f"""
        <div class="metric-card {signal_color}">
            <h3>Action: {signals['action']}</h3>
            <p><strong>Confidence:</strong> {signals['confidence']}</p>
            <p><strong>Entry Price:</strong> {entry_price_str}</p>
            <p><strong>Stop Loss:</strong> {stop_loss_str}</p>
            <p><strong>Target Price:</strong> {target_price_str}</p>
            <p><strong>Risk/Reward:</strong> {signals['risk_reward_ratio'] if signals['risk_reward_ratio'] else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reasoning
        if signals['reasoning']:
            st.subheader("Analysis Reasoning")
            for reason in signals['reasoning']:
                st.markdown(f"• {reason}")
    
    # Support and Resistance levels
    st.subheader("Support & Resistance Levels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Support Level",
            f"Rs {analysis['technical_indicators']['Support']:.2f}",
            delta=f"{((analysis['technical_indicators']['Support'] / analysis['current_price'] - 1) * 100):.2f}%"
        )
    
    with col2:
        st.metric(
            "Current Price",
            f"Rs {analysis['current_price']:.2f}"
        )
    
    with col3:
        st.metric(
            "Resistance Level",
            f"Rs {analysis['technical_indicators']['Resistance']:.2f}",
            delta=f"{((analysis['technical_indicators']['Resistance'] / analysis['current_price'] - 1) * 100):.2f}%"
        )
    
    # Risk Management
    st.subheader("Risk Management")
    if signals['action'] in ['BUY', 'SELL']:
        risk_pct = abs((analysis['current_price'] - signals['stop_loss']) / analysis['current_price'] * 100) if signals['stop_loss'] else 0
        reward_pct = abs((signals['target_price'] - analysis['current_price']) / analysis['current_price'] * 100) if signals['target_price'] else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk %", f"{risk_pct:.2f}%")
        with col2:
            st.metric("Reward %", f"{reward_pct:.2f}%")
        with col3:
            st.metric("R/R Ratio", f"{signals['risk_reward_ratio'] if signals['risk_reward_ratio'] else 'N/A'}")
        
        if risk_pct > 5:
            st.warning("High risk trade - consider reducing position size")
        elif risk_pct < 2:
            st.success("Low risk trade")
        else:
            st.info("Moderate risk trade")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Disclaimer:</strong> This is for educational purposes only. 
    Always do your own research before making investment decisions.</p>
    <p>Data sources: Yahoo Finance, TradingView Technical Analysis, Google News</p>
    <p>ML Models: Random Forest, XGBoost, LightGBM</p>
</div>
""", unsafe_allow_html=True)
