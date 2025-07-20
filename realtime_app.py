import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from enhanced_analyzer import EnhancedStockAnalyzer
from enhanced_visualizer import EnhancedChartVisualizer
from portfolio_manager import PortfolioManager
from realtime_data import RealTimeDataFeed
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Real-Time Stock Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .portfolio-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(90deg, #1e2130 0%, #262730 100%);
        padding: 1rem; border-radius: 0.5rem;
        border-left: 4px solid #00ff00; margin: 0.5rem 0;
    }
    .profit { color: #00ff00; }
    .loss { color: #ff0000; }
    .realtime-data {
        background: linear-gradient(45deg, #2c3e50, #34495e);
        padding: 10px; border-radius: 8px; margin: 5px 0;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def init_components():
    return {
        'analyzer': EnhancedStockAnalyzer(),
        'visualizer': EnhancedChartVisualizer(),
        'portfolio': PortfolioManager(),
        'data_feed': RealTimeDataFeed()
    }

components = init_components()

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['RELIANCE', 'TCS', 'INFY']
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = pd.DataFrame()
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# Header
st.title("📊 Real-Time Stock Trading Platform")
st.markdown("### Advanced Portfolio Management with Live Data & AI Analysis")

# Sidebar
st.sidebar.title("Controls")

# Tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-Time Dashboard", "💼 Portfolio Manager", "🔍 Stock Analysis", "📊 Watchlist"])

with tab1:
    st.header("Real-Time Market Dashboard")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    
    if auto_refresh:
        # Auto-refresh placeholder
        placeholder = st.empty()
        
        # Real-time data display
        col1, col2, col3, col4 = st.columns(4)
        
        # Market overview metrics
        with col1:
            st.metric("Market Status", "OPEN" if datetime.now().hour >= 9 and datetime.now().hour < 16 else "CLOSED")
        
        with col2:
            st.metric("Last Update", st.session_state.last_update.strftime("%H:%M:%S"))
        
        with col3:
            st.metric("Active Stocks", len(st.session_state.watchlist))
        
        with col4:
            portfolio_value = 0
            if not st.session_state.portfolio_data.empty:
                portfolio_value = st.session_state.portfolio_data['Total Value'].sum()
            st.metric("Portfolio Value", f"Rs {portfolio_value:,.2f}")
    
    # Real-time watchlist data
    st.subheader("Live Watchlist Data")
    
    watchlist_data = []
    for symbol in st.session_state.watchlist:
        try:
            # Get real-time data
            ticker = yf.Ticker(symbol + '.NS')
            info = ticker.info
            hist = ticker.history(period='1d', interval='5m')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_close = info.get('previousClose', current_price)
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
                
                watchlist_data.append({
                    'Symbol': symbol,
                    'Current Price': f"Rs {current_price:.2f}",
                    'Change': f"Rs {change:.2f}",
                    'Change %': f"{change_pct:.2f}%",
                    'Volume': f"{hist['Volume'].iloc[-1]:,}",
                    'High': f"Rs {hist['High'].max():.2f}",
                    'Low': f"Rs {hist['Low'].min():.2f}"
                })
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
    
    if watchlist_data:
        df_watchlist = pd.DataFrame(watchlist_data)
        
        # Color code the dataframe based on profit/loss
        def color_change(val):
            if 'Rs -' in val or '-' in val:
                return 'background-color: rgba(255, 0, 0, 0.3)'
            elif 'Rs ' in val and val != 'Rs 0.00':
                return 'background-color: rgba(0, 255, 0, 0.3)'
            return ''
        
        styled_df = df_watchlist.style.applymap(color_change, subset=['Change', 'Change %'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Real-time chart for selected stock
        selected_symbol = st.selectbox("Select stock for real-time chart:", st.session_state.watchlist)
        
        if selected_symbol:
            # Create real-time chart
            ticker = yf.Ticker(selected_symbol + '.NS')
            data = ticker.history(period='1d', interval='5m')
            
            if not data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name=f'{selected_symbol} Price',
                    line=dict(color='#00ff00', width=2)
                ))
                
                fig.update_layout(
                    title=f"Real-Time Price - {selected_symbol}",
                    template='plotly_dark',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Portfolio Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current Portfolio
        st.subheader("Current Portfolio")
        
        portfolio_df = components['portfolio'].get_portfolio()
        
        if not portfolio_df.empty:
            # Calculate current values
            portfolio_analysis = []
            total_investment = 0
            total_current_value = 0
            
            for _, row in portfolio_df.iterrows():
                try:
                    ticker = yf.Ticker(row['symbol'] + '.NS')
                    current_price = ticker.history(period='1d')['Close'].iloc[-1]
                    
                    investment = row['quantity'] * row['buy_price']
                    current_value = row['quantity'] * current_price
                    profit_loss = current_value - investment
                    profit_loss_pct = (profit_loss / investment) * 100
                    
                    total_investment += investment
                    total_current_value += current_value
                    
                    portfolio_analysis.append({
                        'ID': row['id'],
                        'Symbol': row['symbol'],
                        'Quantity': row['quantity'],
                        'Buy Price': f"Rs {row['buy_price']:.2f}",
                        'Current Price': f"Rs {current_price:.2f}",
                        'Investment': f"Rs {investment:.2f}",
                        'Current Value': f"Rs {current_value:.2f}",
                        'P&L': f"Rs {profit_loss:.2f}",
                        'P&L %': f"{profit_loss_pct:.2f}%"
                    })
                except Exception as e:
                    st.error(f"Error processing {row['symbol']}: {e}")
            
            if portfolio_analysis:
                df_portfolio = pd.DataFrame(portfolio_analysis)
                st.dataframe(df_portfolio, use_container_width=True)
                
                # Portfolio summary
                total_pl = total_current_value - total_investment
                total_pl_pct = (total_pl / total_investment) * 100 if total_investment > 0 else 0
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Total Investment", f"Rs {total_investment:.2f}")
                with col_b:
                    st.metric("Current Value", f"Rs {total_current_value:.2f}")
                with col_c:
                    st.metric("Total P&L", f"Rs {total_pl:.2f}", f"{total_pl_pct:.2f}%")
                with col_d:
                    st.metric("No. of Stocks", len(portfolio_analysis))
        else:
            st.info("No stocks in portfolio. Add some stocks below!")
    
    with col2:
        # Add new stock to portfolio
        st.subheader("Add Stock to Portfolio")
        
        with st.form("add_stock_form"):
            new_symbol = st.text_input("Stock Symbol", placeholder="e.g., RELIANCE")
            new_quantity = st.number_input("Quantity", min_value=1, value=1)
            new_buy_price = st.number_input("Buy Price (Rs)", min_value=0.01, value=100.0, step=0.01)
            
            if st.form_submit_button("Add to Portfolio"):
                if new_symbol:
                    components['portfolio'].add_stock(new_symbol.upper(), new_quantity, new_buy_price)
                    st.success(f"Added {new_quantity} shares of {new_symbol.upper()} at Rs {new_buy_price}")
                    st.experimental_rerun()
        
        # Remove stock from portfolio
        st.subheader("Remove Stock")
        portfolio_df = components['portfolio'].get_portfolio()
        if not portfolio_df.empty:
            stock_options = [f"{row['symbol']} (ID: {row['id']})" for _, row in portfolio_df.iterrows()]
            selected_stock = st.selectbox("Select stock to remove:", stock_options)
            
            if st.button("Remove Selected Stock"):
                stock_id = int(selected_stock.split("ID: ")[1].split(")")[0])
                components['portfolio'].remove_stock(stock_id)
                st.success("Stock removed from portfolio")
                st.experimental_rerun()

with tab3:
    st.header("Advanced Stock Analysis")
    
    # Stock analysis input
    analysis_symbol = st.text_input("Enter Stock Symbol for Analysis", value="TCS")
    
    if st.button("Analyze Stock"):
        with st.spinner(f"Analyzing {analysis_symbol}..."):
            try:
                analysis = components['analyzer'].analyze_stock(analysis_symbol)
                if analysis:
                    # Display analysis results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"Rs {analysis['current_price']}")
                    with col2:
                        st.metric("RSI", f"{analysis['technical_indicators']['RSI']:.2f}")
                    with col3:
                        st.metric("Signal", analysis['signals']['action'])
                    with col4:
                        st.metric("Confidence", analysis['signals']['confidence'])
                    
                    # Charts
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Price chart
                        chart = components['visualizer'].create_candlestick_chart(
                            analysis['data'], title=f"{analysis['company_name']} Price Chart"
                        )
                        st.plotly_chart(chart, use_container_width=True)
                    
                    with col_chart2:
                        # Predictions
                        if analysis['predictions']:
                            pred_chart = components['visualizer'].create_prediction_chart(
                                analysis['data'], 
                                analysis['predictions'], 
                                analysis['future_dates'], 
                                analysis['current_price']
                            )
                            st.plotly_chart(pred_chart, use_container_width=True)
                    
                    # Trading signals
                    signals = analysis['signals']
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Trading Signal: {signals['action']}</h3>
                        <p><strong>Confidence:</strong> {signals['confidence']}</p>
                        <p><strong>Entry Price:</strong> Rs {signals['entry_price']:.2f if signals['entry_price'] else 'N/A'}</p>
                        <p><strong>Target:</strong> Rs {signals['target_price']:.2f if signals['target_price'] else 'N/A'}</p>
                        <p><strong>Stop Loss:</strong> Rs {signals['stop_loss']:.2f if signals['stop_loss'] else 'N/A'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if signals['reasoning']:
                        st.subheader("Analysis Reasoning")
                        for reason in signals['reasoning']:
                            st.write(f"• {reason}")
            
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

with tab4:
    st.header("Watchlist Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Watchlist")
        
        if st.session_state.watchlist:
            # Display watchlist with remove buttons
            for i, symbol in enumerate(st.session_state.watchlist):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.write(f"{i+1}. {symbol}")
                with col_b:
                    if st.button(f"Remove", key=f"remove_{symbol}"):
                        st.session_state.watchlist.remove(symbol)
                        st.experimental_rerun()
        else:
            st.info("Watchlist is empty")
    
    with col2:
        st.subheader("Add to Watchlist")
        
        new_watchlist_symbol = st.text_input("Stock Symbol", key="watchlist_input")
        
        if st.button("Add to Watchlist"):
            if new_watchlist_symbol and new_watchlist_symbol.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_watchlist_symbol.upper())
                st.success(f"Added {new_watchlist_symbol.upper()} to watchlist")
                st.experimental_rerun()
            elif new_watchlist_symbol.upper() in st.session_state.watchlist:
                st.warning("Stock already in watchlist")

# Auto-refresh mechanism
if auto_refresh:
    time.sleep(30)
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><strong>Real-Time Stock Trading Platform</strong></p>
    <p>Data sources: Yahoo Finance | Portfolio stored locally</p>
    <p><strong>Disclaimer:</strong> For educational purposes only. Do your research before investing.</p>
</div>
""", unsafe_allow_html=True)
