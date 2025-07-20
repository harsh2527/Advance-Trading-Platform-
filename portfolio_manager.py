import sqlite3
import pandas as pd

class PortfolioManager:
    def __init__(self, db_path='portfolio.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_portfolio_table()

    def create_portfolio_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    buy_price REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

    def add_stock(self, symbol, quantity, buy_price):
        with self.conn:
            self.conn.execute(
                'INSERT INTO portfolio (symbol, quantity, buy_price) VALUES (?, ?, ?)',
                (symbol, quantity, buy_price)
            )

    def remove_stock(self, stock_id):
        with self.conn:
            self.conn.execute('DELETE FROM portfolio WHERE id = ?', (stock_id,))

    def get_portfolio(self):
        df = pd.read_sql('SELECT * FROM portfolio', self.conn)
        return df

    def update_stock(self, stock_id, quantity, new_buy_price):
        with self.conn:
            self.conn.execute(
                'UPDATE portfolio SET quantity = ?, buy_price = ? WHERE id = ?',
                (quantity, new_buy_price, stock_id)
            )
