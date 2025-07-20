import streamlit as st
import pandas as pd
import numpy as np
from stock_analyzer import AdvancedStockAnalyzer
from chart_visualizer import ChartVisualizer
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Trader",
    page_icon="📈",
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = AdvancedStockAnalyzer()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = ChartVisualizer()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Header
st.title("🚀 Advanced Stock Trader")
st.markdown("### AI-Powered Stock Analysis with TradingView Integration")

# Sidebar
st.sidebar.header("🎯 Analysis Controls")

# Stock symbol input
symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="RELIANCE",
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
analyze_button = st.sidebar.button("🔍 Analyze Stock", type="primary")

# Popular stocks quick select
st.sidebar.subheader("📊 Popular Stocks")
popular_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT", "SBIN", "BHARTIARTL"]
selected_popular = st.sidebar.selectbox("Quick Select", [""] + popular_stocks)

if selected_popular:
    symbol = selected_popular

# Main analysis
if analyze_button or st.session_state.analysis_results is not None:
    if analyze_button:
        with st.spinner(f"🔍 Analyzing {symbol}... This may take a few moments."):
            try:
                analysis = st.session_state.analyzer.analyze_stock(symbol)
                st.session_state.analysis_results = analysis
            except Exception as e:
                st.error(f"❌ Error analyzing {symbol}: {str(e)}")
                st.stop()
    
    analysis = st.session_state.analysis_results
    
    if analysis is None:
        st.error("❌ No data available for this symbol. Please check the symbol and try again.")
        st.stop()
    
    # Display results
    col1, col2, col3, col4 = st.columns(4)
    
    # Basic info metrics
    with col1:
        st.metric(
            label="Current Price",
            value=f"₹{analysis['current_price']}",
        )
    
    with col2:
        st.metric(
            label="Market Cap",
            value=f"₹{analysis['market_cap']:,.0f}" if isinstance(analysis['market_cap'], (int, float)) else "N/A"
        )
    
    with col3:
        st.metric(
            label="P/E Ratio",
            value=f"{analysis['pe_ratio']:.2f}" if isinstance(analysis['pe_ratio'], (int, float)) else "N/A"
        )
    
    with col4:
        st.metric(
            label="RSI",
            value=f"{analysis['technical_indicators']['RSI']:.2f}",
            delta="Oversold" if analysis['technical_indicators']['RSI'] < 30 else 
                  "Overbought" if analysis['technical_indicators']['RSI'] > 70 else "Neutral"
        )
    
    # Main content area
    col_main, col_sidebar = st.columns([2, 1])
    
    with col_main:
        # Price chart
        st.subheader("📈 Price Chart & Technical Indicators")
        chart = st.session_state.visualizer.create_candlestick_chart(
            analysis['data'], 
            title=f"{analysis['company_name']} ({symbol})"
        )
        st.plotly_chart(chart, use_container_width=True)
        
        # MACD chart
        st.subheader("📊 MACD Analysis")
        macd_chart = st.session_state.visualizer.create_macd_chart(analysis['data'])
        st.plotly_chart(macd_chart, use_container_width=True)
    
    with col_sidebar:
        # Probability analysis
        st.subheader("🎯 Probability Analysis")
        prob_chart = st.session_state.visualizer.create_probability_chart(analysis['probabilities'])
        st.plotly_chart(prob_chart, use_container_width=True)
        
        # Trading signal
        st.subheader("🚦 Trading Signal")
        signal_chart = st.session_state.visualizer.create_signals_chart(
            analysis['signals'], 
            analysis['current_price']
        )
        st.plotly_chart(signal_chart, use_container_width=True)
    
    # Detailed Analysis Section
    st.header("📋 Detailed Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Probabilities
        st.subheader("📊 Action Probabilities")
        prob_df = pd.DataFrame([analysis['probabilities']]).T
        prob_df.columns = ['Probability (%)']
        prob_df.index = ['Buy', 'Sell', 'Hold']
        
        # Color code the probabilities
        def color_probabilities(val):
            if val.name == 'Buy':
                color = 'background-color: rgba(0, 255, 0, 0.3)'
            elif val.name == 'Sell':
                color = 'background-color: rgba(255, 0, 0, 0.3)'
            else:
                color = 'background-color: rgba(255, 255, 0, 0.3)'
            return [color] * len(val)
        
        styled_prob_df = prob_df.style.apply(color_probabilities, axis=1)
        st.dataframe(styled_prob_df, use_container_width=True)
        
        # Technical indicators
        st.subheader("🔧 Technical Indicators")
        tech_df = pd.DataFrame([analysis['technical_indicators']]).T
        tech_df.columns = ['Value']
        st.dataframe(tech_df, use_container_width=True)
    
    with col2:
        # Trading signals
        st.subheader("🎯 Trading Signals")
        signals = analysis['signals']
        
        # Signal card
        signal_color = "buy-signal" if signals['action'] == 'BUY' else \
                      "sell-signal" if signals['action'] == 'SELL' else "hold-signal"
        
        entry_price_str = f"₹{signals['entry_price']:.2f}" if signals['entry_price'] else 'N/A'
        stop_loss_str = f"₹{signals['stop_loss']:.2f}" if signals['stop_loss'] else 'N/A'
        target_price_str = f"₹{signals['target_price']:.2f}" if signals['target_price'] else 'N/A'
        
        st.markdown(f"""
        <div class="metric-card {signal_color}">
            <h3>🚦 Action: {signals['action']}</h3>
            <p><strong>Confidence:</strong> {signals['confidence']}</p>
            <p><strong>Entry Price:</strong> {entry_price_str}</p>
            <p><strong>Stop Loss:</strong> {stop_loss_str}</p>
            <p><strong>Target Price:</strong> {target_price_str}</p>
            <p><strong>Risk/Reward:</strong> {signals['risk_reward_ratio'] if signals['risk_reward_ratio'] else 'N/A'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Reasoning
        if signals['reasoning']:
            st.subheader("💡 Analysis Reasoning")
            for reason in signals['reasoning']:
                st.markdown(f"• {reason}")
        
        # TradingView summary
        if analysis['tradingview_summary']:
            st.subheader("📺 TradingView Summary")
            tv_summary = analysis['tradingview_summary']
            st.markdown(f"**Recommendation:** {tv_summary['RECOMMENDATION']}")
            st.markdown(f"**Buy Signals:** {tv_summary['BUY']}")
            st.markdown(f"**Sell Signals:** {tv_summary['SELL']}")
            st.markdown(f"**Neutral Signals:** {tv_summary['NEUTRAL']}")
    
    # Support and Resistance levels
    st.subheader("📈 Support & Resistance Levels")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Support Level",
            f"₹{analysis['technical_indicators']['Support']:.2f}",
            delta=f"{((analysis['technical_indicators']['Support'] / analysis['current_price'] - 1) * 100):.2f}%"
        )
    
    with col2:
        st.metric(
            "Current Price",
            f"₹{analysis['current_price']:.2f}"
        )
    
    with col3:
        st.metric(
            "Resistance Level",
            f"₹{analysis['technical_indicators']['Resistance']:.2f}",
            delta=f"{((analysis['technical_indicators']['Resistance'] / analysis['current_price'] - 1) * 100):.2f}%"
        )
    
    # Risk Management
    st.subheader("⚠️ Risk Management")
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
            st.warning("⚠️ High risk trade - consider reducing position size")
        elif risk_pct < 2:
            st.success("✅ Low risk trade")
        else:
            st.info("ℹ️ Moderate risk trade")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. 
    Always do your own research before making investment decisions.</p>
    <p>📊 Data sources: Yahoo Finance, TradingView Technical Analysis</p>
</div>
""", unsafe_allow_html=True)
