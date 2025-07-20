import yfinance as yf
import threading
import time
from datetime import datetime
import pandas as pd

class RealTimeDataFeed:
    def __init__(self):
        self.data_cache = {}
        self.is_running = False
        self.watchlist = []

    def add_to_watchlist(self, symbols):
        if isinstance(symbols, str):
            symbols = [symbols]
        for symbol in symbols:
            if not symbol.endswith('.NS'):
                symbol += '.NS'
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)

    def remove_from_watchlist(self, symbol):
        if not symbol.endswith('.NS'):
            symbol += '.NS'
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)

    def get_real_time_data(self, symbol):
        try:
            if not symbol.endswith('.NS'):
                symbol += '.NS'
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            if not data.empty:
                latest = data.iloc[-1]
                return {
                    'symbol': symbol,
                    'current_price': latest['Close'],
                    'volume': latest['Volume'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
        return None

    def fetch_data_for_watchlist(self):
        while self.is_running:
            for symbol in self.watchlist:
                data = self.get_real_time_data(symbol)
                if data:
                    self.data_cache[symbol] = data
            time.sleep(60)  # Update every minute

    def start_feed(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self.fetch_data_for_watchlist)
            self.thread.daemon = True
            self.thread.start()

    def stop_feed(self):
        self.is_running = False

    def get_cached_data(self, symbol=None):
        if symbol:
            return self.data_cache.get(symbol + '.NS' if not symbol.endswith('.NS') else symbol)
        return self.data_cache
