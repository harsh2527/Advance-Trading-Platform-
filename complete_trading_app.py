import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from advanced_ml_analyzer import AdvancedMLStockAnalyzer
from advanced_visualizer import AdvancedTradingVisualizer
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Trading Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Modern CSS with Glassmorphism and Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #0f0f23 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Enhanced Main Container */
    .main {
        padding: 0rem 1rem;
        backdrop-filter: blur(20px);
    }
    
    /* Modern Sidebar */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.85);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Glassmorphism Cards */
    .metric-row {
        background: rgba(30, 33, 48, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-row:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Premium Trading Cards */
    .trading-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 2rem;
        border-radius: 25px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .trading-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .trading-card:hover::before {
        left: 100%;
    }
    
    .trading-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 30px 80px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Enhanced Prediction Cards */
    .prediction-card {
        background: linear-gradient(45deg, rgba(255, 107, 107, 0.85) 0%, rgba(78, 205, 196, 0.85) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 2rem;
        border-radius: 25px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 20px 60px rgba(255, 107, 107, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    
    .prediction-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 30px 80px rgba(255, 107, 107, 0.3);
        filter: brightness(1.1);
    }
    
    /* Fibonacci Cards with Glow */
    .fibonacci-card {
        background: linear-gradient(135deg, rgba(168, 230, 207, 0.9) 0%, rgba(127, 205, 205, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        color: #1a1a1a;
        box-shadow: 0 15px 50px rgba(168, 230, 207, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 500;
    }
    
    .fibonacci-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 25px 70px rgba(168, 230, 207, 0.3);
        filter: brightness(1.05);
    }
    
    /* Support Resistance with Animation */
    .support-resistance {
        background: linear-gradient(45deg, rgba(255, 217, 61, 0.9) 0%, rgba(255, 107, 107, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 20px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 15px 50px rgba(255, 217, 61, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-weight: 500;
    }
    
    .support-resistance:hover {
        transform: translateY(-3px);
        box-shadow: 0 25px 70px rgba(255, 217, 61, 0.3);
        animation: pulse 2s infinite;
    }
    
    /* Confidence Indicators with Glow */
    .confidence-high { 
        border-left: 5px solid #00FF88;
        box-shadow: -5px 0 15px rgba(0, 255, 136, 0.3);
    }
    .confidence-medium { 
        border-left: 5px solid #FFD700;
        box-shadow: -5px 0 15px rgba(255, 215, 0, 0.3);
    }
    .confidence-low { 
        border-left: 5px solid #FF6B6B;
        box-shadow: -5px 0 15px rgba(255, 107, 107, 0.3);
    }
    
    /* Enhanced Metrics */
    div[data-testid="metric-container"] {
        background: rgba(30, 33, 48, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.3);
    }
    
    /* Tab Enhancement */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 15, 35, 0.4);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.7);
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Button Enhancements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
        filter: brightness(1.1);
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(102, 126, 234, 0); }
        100% { box-shadow: 0 0 0 0 rgba(102, 126, 234, 0); }
    }
    
    @keyframes shimmer {
        0% { background-position: -200px 0; }
        100% { background-position: calc(200px + 100%) 0; }
    }
    
    .loading {
        animation: shimmer 1.5s infinite;
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200px 100%;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(15, 15, 35, 0.3);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #667eea);
    }
    
    /* Title Enhancement */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Sidebar Text */
    .css-1d391kg .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Input Field Enhancement */
    .stTextInput > div > div > input {
        background: rgba(30, 33, 48, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: white;
        font-family: 'Inter', sans-serif;
    }
    
    .stSelectbox > div > div {
        background: rgba(30, 33, 48, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_components():
    return {
        'analyzer': AdvancedMLStockAnalyzer(),
        'visualizer': AdvancedTradingVisualizer()
    }

components = init_components()

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ITC']

# Header
st.title("Advanced Trading Platform")
st.markdown("### Professional Stock Analysis with ML Predictions & Fibonacci Levels")

# Sidebar
st.sidebar.header("Trading Controls")

# Stock input
symbol = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="TCS",
    help="Enter NSE stock symbol (e.g., RELIANCE, TCS, INFY)"
).upper()

# Quick select
popular_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC", "LT", "SBIN", "BHARTIARTL", "WIPRO", "MARUTI"]
selected_stock = st.sidebar.selectbox("Quick Select", [""] + popular_stocks)
if selected_stock:
    symbol = selected_stock

# Analysis button
analyze_button = st.sidebar.button("Analyze Stock", type="primary", use_container_width=True)

# Market status
market_open = 9 <= datetime.now().hour < 16
market_status = "🟢 OPEN" if market_open else "🔴 CLOSED"
st.sidebar.markdown(f"**Market Status:** {market_status}")
st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%H:%M:%S')}")

# Main analysis
if analyze_button:
    with st.spinner(f"Performing Advanced Analysis for {symbol}..."):
        try:
            analysis_result = components['analyzer'].analyze_stock_advanced(symbol)
            if analysis_result:
                st.session_state.analysis_result = analysis_result
                st.success(f"✅ Analysis completed for {symbol}")
            else:
                st.error(f"❌ Could not analyze {symbol}. Please check the symbol.")
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")

# Display analysis results
if st.session_state.analysis_result:
    analysis_result = st.session_state.analysis_result
    
    # Company header
    st.markdown(f"## 📊 {analysis_result['company_name']} ({analysis_result['symbol']})")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price",
            f"₹{analysis_result['current_price']:.2f}",
            help="Live stock price"
        )
    
    with col2:
        if analysis_result.get('predictions') and 'ensemble_prediction' in analysis_result['predictions']:
            day1_pred = analysis_result['predictions']['ensemble_prediction'][0]
            change = ((day1_pred - analysis_result['current_price']) / analysis_result['current_price']) * 100
            st.metric(
                "Tomorrow's Prediction",
                f"₹{day1_pred:.2f}",
                f"{change:+.2f}%",
                help="ML model prediction for next day"
            )
        else:
            st.metric("Tomorrow's Prediction", "N/A")
    
    with col3:
        confidence = analysis_result.get('confidence_score', 0.5)
        st.metric(
            "Confidence Score",
            f"{confidence:.1%}",
            help="Model confidence in predictions"
        )
    
    with col4:
        if isinstance(analysis_result['market_cap'], (int, float)):
            if analysis_result['market_cap'] > 1e12:
                market_cap_str = f"₹{analysis_result['market_cap']/1e12:.1f}T"
            elif analysis_result['market_cap'] > 1e9:
                market_cap_str = f"₹{analysis_result['market_cap']/1e9:.1f}B"
            else:
                market_cap_str = f"₹{analysis_result['market_cap']/1e6:.1f}M"
        else:
            market_cap_str = "N/A"
        
        st.metric("Market Cap", market_cap_str)
    
    with col5:
        pe_ratio = analysis_result.get('pe_ratio', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            st.metric("P/E Ratio", f"{pe_ratio:.2f}")
        else:
            st.metric("P/E Ratio", "N/A")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Technical Analysis", 
        "🎯 Price Predictions", 
        "📐 Fibonacci Levels", 
        "💹 Trading Metrics",
        "🧠 Model Performance"
    ])
    
    with tab1:
        st.subheader("Comprehensive Technical Analysis")
        
        # Create comprehensive chart
        if 'data' in analysis_result:
            comprehensive_chart = components['visualizer'].create_comprehensive_chart(
                analysis_result['data'], 
                analysis_result, 
                f"{analysis_result['company_name']} - Technical Analysis"
            )
            st.plotly_chart(comprehensive_chart, use_container_width=True)
        
        # Technical indicators summary
        st.subheader("Technical Indicators Summary")
        tech_indicators = analysis_result.get('technical_indicators', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rsi = tech_indicators.get('RSI', 'N/A')
            if isinstance(rsi, (int, float)):
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.markdown(f"""
                <div class="metric-row">
                    <h4>RSI</h4>
                    <h2>{rsi:.2f}</h2>
                    <p>{rsi_status}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric-row'><h4>RSI</h4><h2>N/A</h2></div>", unsafe_allow_html=True)
        
        with col2:
            macd = tech_indicators.get('MACD', 'N/A')
            if isinstance(macd, (int, float)):
                macd_status = "Bullish" if macd > 0 else "Bearish"
                st.markdown(f"""
                <div class="metric-row">
                    <h4>MACD</h4>
                    <h2>{macd:.4f}</h2>
                    <p>{macd_status}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric-row'><h4>MACD</h4><h2>N/A</h2></div>", unsafe_allow_html=True)
        
        with col3:
            atr = tech_indicators.get('ATR', 'N/A')
            if isinstance(atr, (int, float)):
                st.markdown(f"""
                <div class="metric-row">
                    <h4>ATR</h4>
                    <h2>{atr:.2f}</h2>
                    <p>Volatility</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric-row'><h4>ATR</h4><h2>N/A</h2></div>", unsafe_allow_html=True)
        
        with col4:
            vol_ratio = tech_indicators.get('Volume_Ratio', 'N/A')
            if isinstance(vol_ratio, (int, float)):
                vol_status = "High" if vol_ratio > 1.5 else "Normal"
                st.markdown(f"""
                <div class="metric-row">
                    <h4>Volume Ratio</h4>
                    <h2>{vol_ratio:.2f}</h2>
                    <p>{vol_status}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='metric-row'><h4>Volume Ratio</h4><h2>N/A</h2></div>", unsafe_allow_html=True)
    
    with tab2:
        st.subheader("7-Day Price Predictions")
        
        if analysis_result.get('predictions'):
            # Create prediction chart
            prediction_chart = components['visualizer'].create_prediction_chart(analysis_result)
            if prediction_chart:
                st.plotly_chart(prediction_chart, use_container_width=True)
            
            # Prediction details
            predictions = analysis_result['predictions']['ensemble_prediction']
            dates = pd.date_range(start=datetime.now(), periods=7, freq='D')
            
            # Create prediction table
            pred_df = pd.DataFrame({
                'Date': dates.strftime('%Y-%m-%d'),
                'Day': [f'Day {i+1}' for i in range(7)],
                'Predicted Price': [f'₹{p:.2f}' for p in predictions],
                'Change from Current': [f'{((p - analysis_result["current_price"]) / analysis_result["current_price"] * 100):+.2f}%' 
                                       for p in predictions]
            })
            
            st.subheader("Detailed Predictions")
            st.dataframe(pred_df, use_container_width=True)
            
            # Trading metrics
            current_price = analysis_result['current_price']
            pred_high = max(predictions)
            pred_low = min(predictions)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>📈 Expected High</h3>
                    <h1>₹{pred_high:.2f}</h1>
                    <p>{((pred_high - current_price) / current_price * 100):+.2f}% upside</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>📉 Expected Low</h3>
                    <h1>₹{pred_low:.2f}</h1>
                    <p>{((pred_low - current_price) / current_price * 100):+.2f}% downside</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                volatility = ((pred_high - pred_low) / current_price) * 100
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>📊 Expected Volatility</h3>
                    <h1>{volatility:.2f}%</h1>
                    <p>Price range variation</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Prediction data not available")
    
    with tab3:
        st.subheader("Fibonacci Analysis")
        
        # Create Fibonacci chart
        if 'data' in analysis_result:
            fib_chart = components['visualizer'].create_fibonacci_levels_chart(
                analysis_result['data'], analysis_result
            )
            st.plotly_chart(fib_chart, use_container_width=True)
        
        # Fibonacci levels table
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📐 Retracement Levels")
            if 'fibonacci_levels' in analysis_result:
                fib_levels = analysis_result['fibonacci_levels']
                for level, price in fib_levels.items():
                    distance = ((price - analysis_result['current_price']) / analysis_result['current_price']) * 100
                    st.markdown(f"""
                    <div class="fibonacci-card">
                        <strong>{level}</strong><br>
                        ₹{price:.2f} ({distance:+.2f}%)
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🎯 Extension Levels")
            if 'fibonacci_extensions' in analysis_result:
                fib_ext = analysis_result['fibonacci_extensions']
                for level, price in fib_ext.items():
                    distance = ((price - analysis_result['current_price']) / analysis_result['current_price']) * 100
                    st.markdown(f"""
                    <div class="fibonacci-card">
                        <strong>{level}</strong><br>
                        ₹{price:.2f} ({distance:+.2f}%)
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Trading Metrics & Risk Analysis")
        
        # Get trading metrics
        metrics = components['visualizer'].create_trading_dashboard_metrics(analysis_result)
        
        # Display key trading metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💹 Price Targets")
            for key in ['Current Price', 'Expected Day 1', 'Expected Day 7', 'Predicted High', 'Predicted Low']:
                if key in metrics:
                    st.markdown(f"**{key}:** {metrics[key]}")
        
        with col2:
            st.markdown("#### ⚖️ Risk Analysis")
            for key in ['Next Resistance', 'Next Support', 'Risk/Reward Ratio', 'Confidence Score', 'Volatility']:
                if key in metrics:
                    st.markdown(f"**{key}:** {metrics[key]}")
        
        # Support and Resistance visualization
        if 'price_targets' in analysis_result:
            tech_levels = analysis_result['price_targets']['technical_levels']
            current_price = analysis_result['current_price']
            
            col1, col2 = st.columns(2)
            
            with col1:
                resistance = tech_levels['resistance']
                upside = ((resistance - current_price) / current_price) * 100
                st.markdown(f"""
                <div class="support-resistance">
                    <h3>🔴 Resistance Level</h3>
                    <h1>₹{resistance:.2f}</h1>
                    <p>{upside:+.2f}% upside potential</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                support = tech_levels['support']
                downside = ((current_price - support) / current_price) * 100
                st.markdown(f"""
                <div class="support-resistance">
                    <h3>🟢 Support Level</h3>
                    <h1>₹{support:.2f}</h1>
                    <p>{downside:.2f}% downside risk</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab5:
        st.subheader("ML Model Performance")
        
        if 'model_performance' in analysis_result:
            # Model performance chart
            model_chart = components['visualizer'].create_model_performance_chart(analysis_result)
            if model_chart:
                st.plotly_chart(model_chart, use_container_width=True)
            
            # Model weights table
            model_weights = analysis_result['model_performance']
            
            st.markdown("#### 🤖 Model Contributions")
            for model, weight in model_weights.items():
                confidence_class = "confidence-high" if weight > 0.25 else "confidence-medium" if weight > 0.15 else "confidence-low"
                st.markdown(f"""
                <div class="metric-row {confidence_class}">
                    <strong>{model}:</strong> {weight:.1%}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Model performance data not available")
    
    # Trading recommendations
    st.markdown("---")
    st.subheader("🎯 Trading Recommendation")
    
    if analysis_result.get('predictions'):
        predictions = analysis_result['predictions']['ensemble_prediction']
        current_price = analysis_result['current_price']
        confidence = analysis_result.get('confidence_score', 0.5)
        
        # Calculate recommendation
        expected_return = ((predictions[-1] - current_price) / current_price) * 100
        
        if expected_return > 3 and confidence > 0.6:
            recommendation = "🟢 STRONG BUY"
            reason = f"Expected 7-day return of {expected_return:+.2f}% with {confidence:.1%} confidence"
        elif expected_return > 1 and confidence > 0.5:
            recommendation = "🟡 BUY"
            reason = f"Moderate upside potential of {expected_return:+.2f}%"
        elif expected_return < -3 and confidence > 0.6:
            recommendation = "🔴 STRONG SELL"
            reason = f"Expected decline of {expected_return:+.2f}% with {confidence:.1%} confidence"
        elif expected_return < -1:
            recommendation = "🟡 SELL"
            reason = f"Potential downside of {expected_return:+.2f}%"
        else:
            recommendation = "⚪ HOLD"
            reason = "Limited price movement expected"
        
        st.markdown(f"""
        <div class="trading-card">
            <h2>{recommendation}</h2>
            <h3>{analysis_result['symbol']}</h3>
            <p>{reason}</p>
            <small>⚠️ This is for educational purposes only. Please do your own research before trading.</small>
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown("""
    ## Advanced Trading Platform!
    
    ### Features:
    - 🤖 **Advanced ML Predictions**: 5 different machine learning models
    - 📐 **Fibonacci Analysis**: Retracement and extension levels
    - 📊 **Technical Indicators**: RSI, MACD, Bollinger Bands, and more
    - 🎯 **Price Targets**: Support, resistance, and predicted levels
    - 📈 **Professional Charts**: Comprehensive trading visualizations
    - 💹 **Risk Analysis**: Risk/reward ratios and confidence scores
    
    ### How to Use:
    1. Enter a stock symbol in the sidebar (e.g., TCS, RELIANCE, INFY)
    2. Click "Analyze Stock" to start the analysis
    3. Explore different tabs for detailed insights
    4. Use the predictions and levels for your trading decisions
    
    **Select a stock from the sidebar to begin!**
    """)
    
    # Quick market overview
    st.subheader("📊 Quick Market Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Top Gainers Today**
        - Stock A: +5.2%
        - Stock B: +3.8%
        - Stock C: +2.1%
        """)
    
    with col2:
        st.markdown("""
        **Top Losers Today**
        - Stock X: -3.5%
        - Stock Y: -2.8%
        - Stock Z: -1.9%
        """)
    
    with col3:
        st.markdown(f"""
        **Market Status**
        - Status: {market_status}
        - Time: {datetime.now().strftime('%H:%M:%S')}
        - Date: {datetime.now().strftime('%Y-%m-%d')}
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>⚠️ Important Disclaimer</strong></p>
    <p>This platform is for educational and to help in research purposes only. 
    All predictions and recommendations are based on historical data and mathematical models.</p>
    <p><strong>Always conduct your own research and consult with financial advisors before making investment decisions.</strong></p>
    <p>Made with ML Models: XGBoost, LightGBM, Random Forest, Neural Networks</p>
</div>
""", unsafe_allow_html=True)
