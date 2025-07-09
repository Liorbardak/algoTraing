import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import copy
from fontTools.misc.cython import returns


class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class TradeOrder:
    """Represents a trading order"""
    ticker: str
    order_type: OrderType
    quantity: float
    price: float
    timestamp: datetime
    order_id: str = None

    def __post_init__(self):
        if self.order_id is None:
            self.order_id = f"{self.ticker}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.order_type.value}"


@dataclass
class Position:
    """Represents a position in a stock"""
    ticker: str
    quantity: float
    current_price: float
    last_updated: datetime

    @property
    def market_value(self) -> float:
        """Calculate current market value of the position"""
        return self.quantity * self.current_price




@dataclass
class PortfolioSnapshot:
    """Represents a snapshot of portfolio state at a specific time"""
    timestamp: datetime
    total_value: float
    cash: float
    positions: Dict[str, Position]
    default_index: Position


class Portfolio:
    """
    A comprehensive portfolio class for trading that manages multiple stocks,
    tracks portions, maintains cash, and handles buy/sell operations.
    """

    def __init__(self,
                 initial_cash: float = 100.0,
                 default_index: str = "S&P 500",
                 portfolio_name: str = "Trading Portfolio"):
        """
        Initialize the portfolio

        Args:
            initial_cash: Starting cash amount
            default_index: Name of the reference index (default: S&P 500)
            portfolio_name: Name of the portfolio
        """
        self.portfolio_name = portfolio_name
        self.initial_cash = initial_cash
        self.cash = initial_cash
  
        self.current_date = datetime.now()

        # Portfolio holdings
        self.positions: Dict[str, Position] = {}

        # History tracking
        self.trade_history: List[TradeOrder] = []
        self.portfolio_history: List[PortfolioSnapshot] = []

        # Reference index 
        self.default_index =  Position(
                ticker=default_index,
                quantity=0,
                current_price=0,
                last_updated=None
            )

        # Performance tracking
        self.total_invested = 0.0
        self.total_realized_pnl = 0.0

    def buy_default_index_with_all_cash(self , default_index_price , date ):
        '''
        Buy default index with all the free cash
        '''
        if self.cash > 0:
            self.default_index.quantity += self.cash / default_index_price
            # Update default
            self.default_index.current_price =  default_index_price
            self.default_index.last_updated = date
            self.cash = 0

    def sell_all_default_index(self , default_index_price , date ):
        '''
        sell all default , get free cash
        '''
        self.cash += self.default_index.quantity*default_index_price
        self.default_index.quantity = 0
        self.default_index.current_price = default_index_price
        self.default_index.last_updated = date


    def is_in(self , ticker: str) -> bool:
        """
        :param ticker: check if ticker is in the portfolio
        :return:
        """
        return (not ticker in self.positions.keys()) or self.positions[ticker].quantity == 0

    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        return self.get_total_invested_stocks_value() + self.get_total_free_value()

    def get_total_invested_stocks_value(self) -> float:
        """Calculate total portfolio invested value (all positions)"""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return positions_value

    def get_total_free_value(self) -> float:
        """cash + default index """
        return self.cash + self.default_index.market_value

    def get_portfolio_weights(self) -> Dict[str, float]:
        """Get the weight/portion of each stock in the portfolio"""
        total_value = self.get_total_value()
        try:
            if total_value == 0:
                return {}
        except:
            total_value = self.get_total_value()

        weights = {}
        for ticker, position in self.positions.items():
            weights[ticker] = position.market_value / total_value

        return weights

    def get_position_summary(self) -> Dict[str, Dict]:
        """Get detailed summary of all positions"""
        summary = {}
        for ticker, position in self.positions.items():
            summary[ticker] = {
                'quantity': position.quantity,
                'current_price': position.current_price,
                'market_value': position.market_value,
                'weight': position.market_value / self.get_total_value() if self.get_total_value() > 0 else 0
            }
        return summary

    def update_prices(self, price_updates:pd.DataFrame,
                      default_index: pd.DataFrame,
                      update_date: pd.Timestamp):
        """
        Update stock prices and reference index value

        Args:
            price_updates: Dict of ticker -> new price
            default_index: New reference index value
            update_date: Date of the update (defaults to current time)
        """

        self.current_date = update_date
        update_tickers = set(price_updates.ticker)

        # Update stock prices
        for ticker in self.positions.keys():
            if ticker in update_tickers:
                self.positions[ticker].current_price = price_updates[price_updates.ticker == ticker].Close.values[0]
                self.positions[ticker].last_updated = update_date

        # Update default index
        self.default_index.current_price = default_index.Close.values[0]
        self.default_index.last_updated = update_date

        # Take a snapshot for history
        self._take_snapshot()

    def buy_stock(self, ticker: str, quantity: float, price: float,
                  timestamp: Optional[datetime] = None) -> bool:
        """
        Buy a stock

        Args:
            ticker: Stock ticker
            quantity: Number of shares to buy
            price: Price per share
            timestamp: Timestamp of the trade

        Returns:
            bool: True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        total_cost = quantity * price

        # # Check if we have enough cash
        # if total_cost > self.cash:
        #     print(f"Insufficient cash. Need ${total_cost:.2f}, have ${self.cash:.2f}")
        #     return False

        # Create trade order
        order = TradeOrder(
            ticker=ticker,
            order_type=OrderType.BUY,
            quantity=quantity,
            price=price,
            timestamp=timestamp
        )

        # Execute the trade
        self.cash -= total_cost
        self.total_invested += total_cost

        # Update or create position
        if ticker in self.positions:
            # Update existing position
            pos = self.positions[ticker]
            total_quantity = pos.quantity + quantity
            pos.quantity = total_quantity
            pos.current_price = price
            pos.last_updated = timestamp
        else:
            # Create new position
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=quantity,
                current_price=price,
                last_updated=timestamp
            )

        # Record the trade
        self.trade_history.append(order)

        print(f"Bought {quantity} shares of {ticker} at ${price:.2f} per share")
        return True

    def sell_stock(self, ticker: str, quantity: float, price: float,
                   timestamp: Optional[datetime] = None  ) -> bool:
        """
        Sell a stock

        Args:
            ticker: Stock ticker
            quantity: Number of shares to sell
            price: Price per share
            timestamp: Timestamp of the trade

        Returns:
            bool: True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Check if we have enough shares
        if ticker not in self.positions or self.positions[ticker].quantity < quantity:
            print(
                f"Insufficient shares. Need {quantity} shares of {ticker}, have {self.positions[ticker].quantity if ticker in self.positions else 0}")
            return False

        # Create trade order
        order = TradeOrder(
            ticker=ticker,
            order_type=OrderType.SELL,
            quantity=quantity,
            price=price,
            timestamp=timestamp
        )

        # Execute the trade
        total_proceeds = quantity * price
        self.cash += total_proceeds

        # Calculate realized P&L
        position = self.positions[ticker]

        # Update position
        position.quantity -= quantity
        position.current_price = price
        position.last_updated = timestamp

        # Remove position if quantity becomes zero
        if position.quantity <= 0:
            del self.positions[ticker]

        # Record the trade
        self.trade_history.append(order)

        return True

    def _take_snapshot(self):
        """Take a snapshot of current portfolio state"""
        snapshot = PortfolioSnapshot(
            timestamp=self.current_date,
            total_value=self.get_total_value(),
            cash=self.cash,
            positions=copy.deepcopy(self.positions),
            default_index=copy.deepcopy(self.default_index),
        )
        self.portfolio_history.append(snapshot)

    def history_to_csv(self, filepath: str):
        """
               Convert portfolio history to data frame and save to file

               Args:
                   filepath: Path to save the file
        """
        # Get all ticker used
        all_tickers = dict()
        for snap in self.portfolio_history:
            all_tickers.update(snap.positions)
        tickers = list(all_tickers.keys())

        all_data = []
        for snap in self.portfolio_history:
            r = {'Date' : pd.Timestamp(snap.timestamp),
                 'total_value': snap.total_value,
                 'cash': snap.cash,
                 'default_index': snap.default_index.market_value,
                 }
            n_tickers = 0
            for ticker in tickers:
                if ticker in snap.positions.keys():
                    if snap.positions[ticker].market_value > 0:
                        n_tickers += 1
                    r[ticker] = snap.positions[ticker].market_value
                else:
                    r[ticker] = 0
            r['n_ticker_in_protofolio'] = n_tickers
            all_data.append(r)
        pd.DataFrame(all_data).to_csv(filepath)

    def save_history(self, filepath: str, format: str = 'pickle'):
        """
        Save portfolio history to file

        Args:
            filepath: Path to save the file
            format: 'pickle' or 'json'
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if format.lower() == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'portfolio_history': self.portfolio_history,
                    'trade_history': self.trade_history,
                    'default_index_history': self.default_index_history,
                    'portfolio_name': self.portfolio_name,
                    'initial_cash': self.initial_cash
                }, f)
        elif format.lower() == 'json':
            # Convert to JSON-serializable format
            data = {
                'portfolio_history': [
                    {
                        'timestamp': snap.timestamp.isoformat(),
                        'total_value': snap.total_value,
                        'cash': snap.cash,
                        'positions': {
                            ticker: {
                                'quantity': pos.quantity,
                                'current_price': pos.current_price,
                                'last_updated': pos.last_updated.isoformat()
                            } for ticker, pos in snap.positions.items()
                        },
                        'default_index_value': snap.default_index_value,
                        'default_index_name': snap.default_index_name
                    } for snap in self.portfolio_history
                ],
                'trade_history': [
                    {
                        'ticker': trade.ticker,
                        'order_type': trade.order_type.value,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'timestamp': trade.timestamp.isoformat(),
                        'order_id': trade.order_id
                    } for trade in self.trade_history
                ],
                'default_index_history': [
                    {'timestamp': ts.isoformat(), 'value': val}
                    for ts, val in self.default_index_history
                ],
                'portfolio_name': self.portfolio_name,
                'initial_cash': self.initial_cash
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

        print(f"Portfolio history saved to {filepath}")

    def load_history(self, filepath: str, format: str = 'pickle'):
        """
        Load portfolio history from file

        Args:
            filepath: Path to the history file
            format: 'pickle' or 'json'
        """
        if not os.path.exists(filepath):
            print(f"History file {filepath} not found")
            return

        if format.lower() == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif format.lower() == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Convert JSON data back to objects
            self.portfolio_history = []
            for snap_data in data['portfolio_history']:
                positions = {}
                for ticker, pos_data in snap_data['positions'].items():
                    positions[ticker] = Position(
                        ticker=ticker,
                        quantity=pos_data['quantity'],
                        current_price=pos_data['current_price'],
                        last_updated=datetime.fromisoformat(pos_data['last_updated'])
                    )

                snapshot = PortfolioSnapshot(
                    timestamp=datetime.fromisoformat(snap_data['timestamp']),
                    total_value=snap_data['total_value'],
                    cash=snap_data['cash'],
                    positions=positions,
                    default_index_value=snap_data['default_index_value'],
                    default_index_name=snap_data['default_index_name']
                )
                self.portfolio_history.append(snapshot)

            self.trade_history = []
            for trade_data in data['trade_history']:
                trade = TradeOrder(
                    ticker=trade_data['ticker'],
                    order_type=OrderType(trade_data['order_type']),
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    timestamp=datetime.fromisoformat(trade_data['timestamp']),
                    order_id=trade_data['order_id']
                )
                self.trade_history.append(trade)

            self.default_index_history = [
                (datetime.fromisoformat(item['timestamp']), item['value'])
                for item in data['default_index_history']
            ]

            self.portfolio_name = data['portfolio_name']
            self.initial_cash = data['initial_cash']

            # Restore current state from latest snapshot
            if self.portfolio_history:
                latest = self.portfolio_history[-1]
                self.cash = latest.cash
                self.positions = latest.positions.copy()
                self.current_date = latest.timestamp
                self.default_index_value = latest.default_index_value

        print(f"Portfolio history loaded from {filepath}")

    def get_portfolio_summary(self) -> str:
        """Get a formatted summary of the portfolio"""
        metrics = self.get_performance_metrics()
        weights = self.get_portfolio_weights()

        summary = f"""
=== {self.portfolio_name} Portfolio Summary ===
Date: {self.current_date.strftime('%Y-%m-%d %H:%M:%S')}

Total Value: ${metrics.get('total_value', 0):,.2f}
Cash: ${metrics.get('cash', 0):,.2f}
Invested Value: ${metrics.get('invested_value', 0):,.2f}

Performance:
- Total Return: ${metrics.get('total_return', 0):,.2f} ({metrics.get('total_return_percent', 0):.2f}%)
- Realized P&L: ${metrics.get('realized_pnl', 0):,.2f}
- Unrealized P&L: ${metrics.get('unrealized_pnl', 0):,.2f}
- Reference Index Return: {metrics.get('default_index_return_percent', 0):.2f}%
- Excess Return: {metrics.get('excess_return', 0):.2f}%

Positions ({metrics.get('num_positions', 0)} stocks):
"""

        for ticker, weight in weights.items():
            if ticker != 'CASH':
                position = self.positions[ticker]
                summary += f"- {ticker}: {weight:.1%} (${position.market_value:,.2f})\n"

        summary += f"Cash: {weights.get('CASH', 0):.1%}\n"
        summary += f"\nTotal Trades: {metrics.get('num_trades', 0)}"

        return summary

    def plot_portfolio_history(self, save_path: Optional[str] = None):
        """Plot portfolio performance over time"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
        except ImportError:
            print("matplotlib not available for plotting")
            return

        if not self.portfolio_history:
            print("No portfolio history to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Portfolio value over time
        dates = [snap.timestamp for snap in self.portfolio_history]
        values = [snap.total_value for snap in self.portfolio_history]
        cash_values = [snap.cash for snap in self.portfolio_history]

        ax1.plot(dates, values, label='Total Portfolio Value', linewidth=2)
        ax1.plot(dates, cash_values, label='Cash', alpha=0.7)
        ax1.axhline(y=self.initial_cash, color='red', linestyle='--', alpha=0.5, label='Initial Investment')
        ax1.set_title(f'{self.portfolio_name} - Portfolio Value Over Time')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Reference index comparison
        if self.default_index_history:
            ref_dates = [ts for ts, _ in self.default_index_history]
            ref_values = [val for _, val in self.default_index_history]

            # Normalize reference index to same scale as portfolio
            ref_normalized = [val / ref_values[0] * self.initial_cash for val in ref_values]

            ax2.plot(ref_dates, ref_normalized, label=f'{self.default_index}', alpha=0.7)
            ax2.plot(dates, values, label='Portfolio', linewidth=2)
            ax2.set_title('Portfolio vs Reference Index')
            ax2.set_ylabel('Value ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio chart saved to {save_path}")

        plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Create a portfolio
    portfolio = Portfolio(initial_cash=100000, portfolio_name="Test Portfolio")

    # Simulate some trades
    portfolio.buy_stock("AAPL", 100, 150.0)
    portfolio.buy_stock("MSFT", 50, 300.0)
    portfolio.buy_stock("GOOGL", 25, 2800.0)

    # Update prices
    portfolio.update_prices({
        "AAPL": 155.0,
        "MSFT": 310.0,
        "GOOGL": 2850.0
    }, default_index_value=105.0)

    # Sell some shares
    portfolio.sell_stock("AAPL", 25, 155.0)

    # Print summary
    print(portfolio.get_portfolio_summary())

    # Save history
    portfolio.save_history("test_portfolio_history.pkl")

    # Create new portfolio and load history
    new_portfolio = Portfolio()
    new_portfolio.load_history("test_portfolio_history.pkl")
    print("\nLoaded portfolio summary:")
    print(new_portfolio.get_portfolio_summary())