# Advanced Trading Platform

A professional, advanced stock trading analysis platform built using Python and Streamlit. This platform integrates machine learning models, technical analysis indicators, Fibonacci analysis, real-time data feeds, and comprehensive visualizations to provide actionable trading insights.

---

## Features

- **Machine Learning Predictions**: Ensemble of Random Forest, XGBoost, LightGBM, Gradient Boosting, and Neural Networks to predict stock price movements.
- **Technical Analysis**: Multiple indicators including RSI, MACD, ATR, Bollinger Bands, and Fibonacci retracements and extensions.
- **Fibonacci Analysis**: Calculations and visualizations of retracement and extension levels for price predictions.
- **Real-Time Data**: Live stock price updates and news sentiment integration.
- **Comprehensive Charts**: Interactive Plotly charts for price, technical indicators, and predictions.
- **Portfolio Management**: Watchlists and portfolio tracking via SQLite backend.
- **Risk & Trading Metrics**: Confidence scores, support and resistance levels, risk/reward ratios.
- **Beautiful UI**: Modern dark theme with glassmorphism styling and smooth animations using custom CSS.

---

## Project Structure

| File Name                 | Description                                                  |
|---------------------------|--------------------------------------------------------------|
| `complete_trading_app.py` | Main Streamlit app managing UI and user interaction          |
| `advanced_ml_analyzer.py` | Implements machine learning models, feature engineering, predictions |
| `advanced_visualizer.py`  | Visualization utilities to generate charts and dashboards    |
| `portfolio_manager.py`    | Manages portfolios, watchlists, and persistence               |
| `realtime_app.py`         | Real-time stock price dashboard and updates                    |
| `realtime_data.py`        | Fetches and processes live market data                        |
| `chart_visualizer.py`     | Additional charting helper functions                          |
| `enhanced_analyzer.py`    | Experimental ML model improvements                            |
| `enhanced_visualizer.py`  | Enhanced visualizations for improved UI/UX                   |
| `app_*.py`               | Various app iterations and improvements                       |
| `requirements.txt`        | Python dependencies required to run the application          |

---

## Installation

### Prerequisites

- Python 3.10 or newer
- Git (optional, for cloning repository)

### Setup instructions

1. Clone or download the repository and navigate to the project directory:

```bash
cd D:\AI\mark\advanced_trader
```

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

3. Install required Python packages:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run complete_trading_app.py
```

---

## Usage

- Use the sidebar to enter NSE stock symbols or select from the quick select list.
- Click "Analyze Stock" to perform advanced ML-based analysis.
- Explore the tabs for technical indicators, price predictions, Fibonacci levels, trading metrics, and model performance.
- Use the watchlist feature to save your favorite stocks.
- View real-time price updates and news sentiment in dedicated dashboards.

---

## Troubleshooting

### "Cannot analyze symbol" Error

If you encounter this error, try the following:

1. **Verify Symbol**: Ensure the stock symbol is correct and exists on Yahoo Finance
2. **Check Data Availability**: Some stocks may have insufficient historical data
3. **Network Issues**: Verify your internet connection for API calls
4. **Symbol Format**: Use the correct format (e.g., "RELIANCE.NS" for NSE stocks)

### Common Solutions

- Try popular symbols like RELIANCE.NS, TCS.NS, INFY.NS first
- Ensure at least 100 days of historical data are available
- Check if the market is open (some real-time features require active trading)

---

## Dependencies

- `streamlit`: Web app framework for Python
- `yfinance`: Yahoo Finance data retrieval
- `pandas`, `numpy`: Data processing and numerical operations
- `scikit-learn`, `xgboost`, `lightgbm`: ML libraries for model training and prediction
- `ta`: Technical analysis indicators
- `plotly`: Interactive visualization library
- `requests`, `beautifulsoup4`: Web scraping and API requests
- `textblob`: Sentiment analysis
- `tradingview-ta`: Technical analysis API
- `apscheduler`: Job scheduling for real-time updates

---

## Architecture

The platform follows a modular architecture:

1. **Data Layer**: `realtime_data.py` handles data fetching from various sources
2. **Analysis Layer**: `advanced_ml_analyzer.py` processes data and generates predictions
3. **Visualization Layer**: `advanced_visualizer.py` creates interactive charts
4. **Application Layer**: `complete_trading_app.py` manages the user interface
5. **Persistence Layer**: `portfolio_manager.py` handles data storage

---

## Machine Learning Models

The platform uses an ensemble approach with the following models:

- **Random Forest**: Robust against overfitting, handles feature importance well
- **XGBoost**: Gradient boosting with excellent performance on structured data
- **LightGBM**: Fast gradient boosting framework
- **Gradient Boosting**: Traditional boosting method for comparison
- **Neural Network**: Deep learning approach for complex pattern recognition

Each model contributes to the final prediction through weighted voting.

---

## Technical Indicators

The platform calculates and visualizes:

- **Momentum**: RSI, MACD, Stochastic Oscillator
- **Trend**: Moving Averages, ADX, Parabolic SAR
- **Volatility**: Bollinger Bands, ATR, Volatility Index
- **Volume**: OBV, Volume SMA, VWAP
- **Fibonacci**: Retracement and Extension levels

---

## Author

**Harsh Malhotra**

An avid trader and developer, passionate about applying machine learning and data science in financial markets to build useful tools and insights.

---

## License & Disclaimer

This project is for educational and informational purposes only. It is not financial advice. Trading in financial markets involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk after due diligence and consulting with financial professionals.

---

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

## Contact

For feedback, suggestions, or questions, please reach out to Harsh Malhotra.

---

*Last Updated: July 2025*
