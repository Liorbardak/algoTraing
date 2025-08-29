"""
Trading Policy Module

This module implements a flexible trading policy framework using the Factory pattern.
It provides:
- A registry system for different trading strategies
- Base class for all trading policies
- Concrete implementations of specific trading strategies
- Portfolio management integration
- Ticker scoring and selection algorithms

The framework supports multiple trading strategies that can be dynamically created
and configured through the configuration system.

Author: dadabardak
Last updated: 2024-03-29
Version: 1.0
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Set, Any, Optional, Tuple
from datetime import datetime
from copy import copy

from config.config import ConfigManager
from utils.protofolio import Portfolio


class TradingPolicy:
    """
    Base class for all trading policies using the Factory pattern.

    This class provides a registry system that allows different trading strategies
    to be registered and instantiated dynamically by name. This design enables
    easy extension of the system with new trading policies without modifying
    existing code.

    The registry pattern allows for:
    - Dynamic policy creation based on configuration
    - Easy addition of new trading strategies
    - Centralized policy management
    - Type-safe policy instantiation

    Class Attributes:
        _registry (Dict[str, type]): Registry of all available trading policies
    """

    # Class-level registry to store all available trading policies
    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a trading policy class in the registry.

        This decorator allows trading policy classes to be automatically
        registered with a given name, making them available for dynamic
        instantiation through the create() method.

        Args:
            name (str): Unique identifier for the trading policy

        Returns:
            function: Decorator function that registers the policy class

        Example:
            @TradingPolicy.register("MyStrategy")
            class MyTradingPolicy(TradingPolicy):
                pass
        """

        def decorator(policy_class: type) -> type:
            cls._registry[name] = policy_class
            return policy_class

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> 'TradingPolicy':
        """
        Factory method to create trading policy instances by name.

        This method looks up the requested policy in the registry and
        instantiates it with the provided arguments. This allows for
        dynamic policy creation based on configuration.

        Args:
            name (str): Name of the trading policy to create
            *args: Positional arguments to pass to the policy constructor
            **kwargs: Keyword arguments to pass to the policy constructor

        Returns:
            TradingPolicy: Instance of the requested trading policy

        Raises:
            ValueError: If the requested policy name is not registered

        Example:
            policy = TradingPolicy.create("MostBasic", config, default_index_name='snp')
        """
        if name not in cls._registry:
            available_policies = list(cls._registry.keys())
            raise ValueError(
                f"Unknown policy: '{name}'. "
                f"Available policies: {available_policies}"
            )

        # Instantiate and return the requested policy
        return cls._registry[name](*args, **kwargs)

    @classmethod
    def list_available_policies(cls) -> list:
        """
        Get a list of all registered trading policies.

        Returns:
            list: List of registered policy names
        """
        return list(cls._registry.keys())


@TradingPolicy.register("MostBasic")
class TradingPolicyMostBasic(TradingPolicy):
    """
    Basic momentum-based trading strategy that combines analyst recommendations
    with technical indicators to make buy/sell decisions.

    This strategy uses:
    - Analyst complement data (recommendations) for fundamental analysis
    - Moving averages (MA150, MA200) for technical analysis
    - Portfolio weight management for risk control
    - Momentum indicators (price above moving averages, positive slope)
    """

    def __init__(self, config, default_index_name='snp'):
        """
        Initialize the trading policy with configuration parameters.

        Args:
            config: Configuration object containing trading parameters
            default_index_name (str): Name of the default index (e.g., 'snp' for S&P 500)
        """
        self.name = "MostBasic"
        self.config = config
        self.portfolio = Portfolio()
        self.default_index_name = default_index_name

    def score_tickers(self, date, tickers_df, complement_df):
        """
        Score all available tickers for potential trading on a given date.

        This method evaluates both existing portfolio holdings and new potential
        investments based on multiple criteria including analyst recommendations,
        technical indicators, and portfolio constraints.

        Args:
            date: Trading date to evaluate
            tickers_df: DataFrame containing historical price and technical indicator data
            complement_df: DataFrame containing analyst recommendation data

        Returns:
            dict: Dictionary of ticker scores with detailed scoring breakdown
        """
        # Get current portfolio composition
        portfolio_weights = self.portfolio.get_portfolio_weights()
        tickers_score = {}

        # ====================================================================
        # RE-EVALUATE EXISTING PORTFOLIO HOLDINGS
        # ====================================================================
        # Score tickers already in the portfolio to determine if we should
        # maintain, increase, or decrease positions
        for ticker, portfolio_weight in portfolio_weights.items():
            score = self.score_ticker(
                date, ticker, tickers_df, complement_df, portfolio_weight
            )

            if score['weighted_score'] > 0:
                # Only keep tickers with positive scores (valid for buying/holding)
                tickers_score[ticker] = score

        # ====================================================================
        # EVALUATE NEW POTENTIAL INVESTMENTS
        # ====================================================================
        # Find tickers not currently in portfolio that are available for trading :
           # tickers that got complement in current date or in the previous quarter
        available_tickers = set(
            complement_df[(complement_df.Date.dt.normalize() >= (date - pd.Timedelta(days=90))) &
            (complement_df.Date.dt.normalize() <= date) ].ticker
        )

        tickers_not_in_portfolio = available_tickers - set(portfolio_weights.keys())

        # Score each potential new investment
        for ticker in tickers_not_in_portfolio:
            score = self.score_ticker(date, ticker, tickers_df, complement_df, 0)

            if score['weighted_score'] > 0:
                # Only consider tickers with positive scores for investment
                tickers_score[ticker] = score

        return tickers_score

    def score_ticker(self, date, ticker, tickers_df, complement_df, portfolio_weight):
        """
        Comprehensive scoring system for individual tickers based on multiple factors.

        The scoring considers:
        1. Portfolio-based constraints (position size limits)
        2. Analyst complement data (recommendation quality and recency)
        3. Technical price indicators (moving averages, momentum)

        Args:
            date: Date for evaluation
            ticker: Stock ticker symbol
            tickers_df: Price and technical data
            complement_df: Analyst recommendation data
            portfolio_weight: Current weight of ticker in portfolio (0 if not held)

        Returns:
            dict: Detailed scoring breakdown with weighted final score
        """
        # Initialize scoring structure with all components
        ticker_score = {
            'weighted_score': 0,
            'price_based_score': 0,
            'complement_based_score': 0,
            'portfolio_based_score': 0,
            'portfolio_weight': portfolio_weight
        }

        # ====================================================================
        # PORTFOLIO-BASED SCORING (Risk Management)
        # ====================================================================
        # Check if current position size exceeds maximum allowed per ticker
        max_portion_per_ticker = self.config.get_parameter('portfolio', 'max_portion_per_ticker')

        if portfolio_weight > max_portion_per_ticker:
            # Position is too large - reject immediately to prevent overconcentration
            return ticker_score
        else:
            # Position size is acceptable - pass this criterion
            # TODO: Implement graduated scoring based on position size
            ticker_score['portfolio_based_score'] = 100

        # ====================================================================
        # ANALYST COMPLEMENT-BASED SCORING (Fundamental Analysis)
        # ====================================================================
        # Find all analyst recommendations for this ticker up to current date
        complements_before_current_time = complement_df[
            (complement_df.ticker == ticker) &
            (complement_df.Date.values.astype('datetime64[D]') <= np.datetime64(date))
            ]

        if len(complements_before_current_time) == 0:
            # No analyst coverage available - too risky to invest
            return ticker_score

        # Get the most recent analyst recommendations
        latest_complements = complements_before_current_time[
            complements_before_current_time.Date == complements_before_current_time.Date.max()
            ]

        last_complement_date = pd.Timestamp(latest_complements.Date.values[0], tz='UTC')

        # Check if recommendations are recent enough to be relevant
        if (date - last_complement_date) > pd.Timedelta(days=90):
            # Recommendations are stale (>90 days old) - don't rely on them
            # TODO: Consider graduated scoring based on recency
            return ticker_score

        # Extract analyst recommendation metrics
        positive_analysts = latest_complements.number_of_analysts_comp.values[0]
        total_analysts = latest_complements.total_number_of_analysts.values[0]

        # Get configuration thresholds for buy decisions
        min_complements_th1 = self.config.get_parameter('buy', 'min_complements_th1')
        min_complements_th2 = self.config.get_parameter('buy', 'min_complements_th2')
        complements_portion_th2 = self.config.get_parameter('buy', 'complements_portion_th2')

        # Calculate analyst consensus strength
        positive_analyst_ratio = positive_analysts / (total_analysts + 1e-6)
        strong_consensus = positive_analyst_ratio > complements_portion_th2

        # Determine if analyst recommendations meet buy criteria
        # Two paths: either strong consensus OR high absolute number of recommendations
        complement_buy_signal = (
                ((positive_analysts >= min_complements_th2) & strong_consensus) |
                (positive_analysts >= min_complements_th1)
        )

        if not complement_buy_signal:
            # Analyst recommendations don't meet our criteria
            return ticker_score
        else:
            # Analyst recommendations support buying
            # TODO: Implement graduated scoring based on recommendation strength
            ticker_score['complement_based_score'] = 100

        # ====================================================================
        # TECHNICAL PRICE-BASED SCORING (Technical Analysis)
        # ====================================================================
        # Get historical price data for technical analysis
        ticker_price_data = tickers_df[
            (tickers_df.ticker == ticker) & (tickers_df.Date <= date)
            ]

        if(len(ticker_price_data) == 0):
            print(f'no data for {ticker} in {date} ')
            return ticker_score
        # Check moving average trend (momentum indicator)
        ma_150_slope_positive = ticker_price_data.ma_150_slop.values[-1] > 0


        # Price position relative to moving averages (trend strength)
        current_price = ticker_price_data.Close.values[-1]
        ma_150_current = ticker_price_data.ma_150.values[-1]
        ma_200_current = ticker_price_data.ma_200.values[-1]

        # Different criteria based on how recent the analyst recommendations are
        if (date - last_complement_date) < pd.Timedelta(days=2):
            # Recent recommendations: Check if price is above MA150 today
            price_above_ma150 = ma_150_current < current_price
        else:
            # Older recommendations: Require sustained strength (10-day period above MA150)
            price_above_ma150 = np.all(
                ticker_price_data.ma_150.values[-10:] < ticker_price_data.Close.values[-10:]
            )

        # Long-term trend confirmation: price above 200-day moving average
        price_above_ma200 = ma_200_current < current_price

        # Combine all technical conditions for buy signal
        technical_buy_signal = (
                price_above_ma200 &
                price_above_ma150 &
                ma_150_slope_positive
        )

        if technical_buy_signal:
            # Technical indicators support buying
            # TODO: Implement graduated scoring based on technical strength
            ticker_score['price_based_score'] = 100

        # ====================================================================
        # FINAL SCORE CALCULATION
        # ====================================================================
        # Use minimum score across all categories (all must pass)
        # This ensures we only buy when ALL criteria are met
        # TODO: Consider weighted average instead of minimum for more nuanced scoring
        ticker_score['weighted_score'] = np.min([
            ticker_score['portfolio_based_score'],
            ticker_score['price_based_score'],
            ticker_score['complement_based_score']
        ])

        return ticker_score

    def buy(self, date, tickers_score, tickers_df, default_index_df):
        """
        Execute buy decisions based on ticker scores and portfolio constraints.

        This method:
        1. Determines how many new positions to add
        2. Calculates appropriate position sizes
        3. Rebalances existing positions
        4. Executes all buy/sell orders

        Args:
            date: Trading date
            tickers_score: Dictionary of ticker scores from score_tickers()
            tickers_df: Current price data for order execution
            default_index_df: Default index data for reference
        """
        # Get current portfolio state
        current_positions = len(self.portfolio.positions)
        current_portfolio_weights = self.portfolio.get_portfolio_weights()

        # Identify potential new investments
        new_tickers = list(set(tickers_score.keys()) - set(self.portfolio.positions.keys()))

        # Calculate how many new positions we can add
        max_total_positions = self.config.get_parameter('portfolio', 'max_number_of_tickers')
        available_position_slots = max_total_positions - current_positions
        new_positions_to_add = min(available_position_slots, len(new_tickers))

        # ====================================================================
        # DETERMINE NEW PORTFOLIO COMPOSITION
        # ====================================================================
        new_portfolio_weights = copy(current_portfolio_weights)

        if new_positions_to_add > 0:
            # Select new tickers to add
            # TODO: Prioritize by weighted_score instead of arbitrary selection
            new_tickers_to_add = new_tickers[:new_positions_to_add]

            if current_positions > 0:
                # Calculate position size for new holdings based on current portfolio
                average_position_size = sum(current_portfolio_weights.values()) / len(current_portfolio_weights)
                max_position_size = self.config.get_parameter('portfolio', 'max_portion_per_ticker')
                new_position_size = min(average_position_size, max_position_size)

                # Add new positions to portfolio weights
                for ticker in new_tickers_to_add:
                    new_portfolio_weights[ticker] = new_position_size
            else:
                # First positions in empty portfolio
                initial_position_size = self.config.get_parameter('portfolio', 'max_portion_per_ticker')
                new_portfolio_weights = {ticker: initial_position_size for ticker in new_tickers_to_add}

        # ====================================================================
        # ENFORCE POSITION SIZE LIMITS
        # ====================================================================
        max_position_size = self.config.get_parameter('portfolio', 'max_portion_per_ticker')
        for ticker in new_portfolio_weights.keys():
            new_portfolio_weights[ticker] = min(new_portfolio_weights[ticker], max_position_size)

        # ====================================================================
        # NORMALIZE PORTFOLIO WEIGHTS
        # ====================================================================
        total_weights = sum(new_portfolio_weights.values())

        if total_weights > 1.0:
            # Normalize to prevent over-allocation
            new_portfolio_weights = {
                ticker: weight / total_weights
                for ticker, weight in new_portfolio_weights.items()
            }
        elif total_weights < 1.0:
            # There's free cash available
            # TODO: Implement logic to allocate remaining cash
            free_cash_weight = 1.0 - total_weights
            # Could distribute equally among existing positions or hold as cash

        # ====================================================================
        # EXECUTE PORTFOLIO REBALANCING
        # ====================================================================
        total_portfolio_value = self.portfolio.get_total_value()

        for ticker, target_weight in new_portfolio_weights.items():
            # Get current price for trade execution
            current_price = tickers_df[tickers_df.ticker == ticker].Close.values[0]

            if ticker not in current_portfolio_weights:
                # NEW POSITION: Calculate shares to buy
                target_value = total_portfolio_value * target_weight
                shares_to_buy = target_value / current_price
                self.portfolio.buy_stock(ticker, shares_to_buy, current_price, date)

            else:
                # EXISTING POSITION: Rebalance (buy more or sell some)
                current_weight = current_portfolio_weights[ticker]
                weight_change = target_weight - current_weight
                value_change = total_portfolio_value * weight_change

                if value_change > 0:
                    # Increase position
                    shares_to_buy = value_change / current_price
                    self.portfolio.buy_stock(ticker, shares_to_buy, current_price, date)
                else:
                    # Decrease position
                    shares_to_sell = abs(value_change) / current_price
                    self.portfolio.sell_stock(ticker, shares_to_sell, current_price, date)

        # ====================================================================
        # PORTFOLIO VALIDATION
        # ====================================================================
        # Debug output for monitoring
        print(f"Portfolio on {date}: {len(self.portfolio.get_portfolio_weights())}  {self.portfolio.get_portfolio_weights()}")

        # Ensure we haven't created a negative cash position
        assert self.portfolio.cash >= -1e-6, (
            f"Negative cash position detected: {self.portfolio.cash} on {date}"
        )

    def sell(self, date, ticker, tickers_df, complement_df, portfolio_weight):
        """
        Evaluate whether to sell a ticker based on technical exit criteria.

        This method implements a simple moving average crossover exit strategy:
        - Sell when price closes below MA200 for 2 consecutive days

        Args:
            date: Current trading date
            ticker: Stock ticker to evaluate for selling
            tickers_df: Historical price data
            complement_df: Analyst data (not used in current implementation)
            portfolio_weight: Current portfolio weight (not used in current implementation)
        """
        if not self.portfolio.is_in(ticker):
            # Not holding this ticker - nothing to sell
            return

        # ====================================================================
        # TECHNICAL EXIT CRITERIA
        # ====================================================================
        # Get historical price data for this ticker
        ticker_data = tickers_df[tickers_df.ticker == ticker].sort_values('Date')
        prices = ticker_data.Close.values
        ma_200 = ticker_data.ma_200.values

        # Find current date index in the data
        current_index = np.where(ticker_data.Date == date)[0][0]

        # Exit condition: Price below MA200 for 2 consecutive days
        # This helps avoid whipsaws while still providing timely exits
        price_below_ma200_today = (
                ~np.isnan(ma_200[current_index]) &
                (ma_200[current_index] > prices[current_index])
        )
        price_below_ma200_yesterday = (
                ma_200[current_index - 1] > prices[current_index - 1]
        )

        sell_signal = price_below_ma200_today & price_below_ma200_yesterday

        if sell_signal:
            # Execute complete exit from this position
            current_price = prices[current_index]
            shares_to_sell = self.portfolio.positions[ticker].quantity
            self.portfolio.sell_stock(ticker, shares_to_sell, current_price, date)

    def trade_recurrent(self, date, tickers, tickers_df, complement_df, default_index):
        """
        Execute the complete trading workflow for a single trading day.

        This is the main orchestration method that coordinates:
        1. Portfolio updates with current prices
        2. Liquidation of default index holdings for trading capital
        3. Exit evaluation for existing positions
        4. Entry evaluation for new and existing positions
        5. Reinvestment of remaining cash in default index

        Args:
            date: Current trading date
            tickers: List of all tradeable tickers
            tickers_df: Historical price and technical data
            complement_df: Analyst recommendation data
            default_index: Default index (e.g., S&P 500) data for cash management
        """
        # ====================================================================
        # DAILY PORTFOLIO MAINTENANCE
        # ====================================================================
        # Update all position values with current market prices
        daily_ticker_data = tickers_df[tickers_df.Date.dt.normalize() == date]
        daily_index_data = default_index[default_index.Date.dt.normalize() == date]

        self.portfolio.update_prices(daily_ticker_data, daily_index_data, date)

        # Get current portfolio composition before trading
        current_portfolio_weights = self.portfolio.get_portfolio_weights()

        # ====================================================================
        # CASH MANAGEMENT: LIQUIDATE DEFAULT INDEX HOLDINGS
        # ====================================================================
        # Convert default index holdings to cash for active trading
        # This cash will be reinvested at the end of the trading day
        current_index_price = daily_index_data.Close.values[0]
        self.portfolio.sell_all_default_index(current_index_price, date)

        # ====================================================================
        # EXIT MANAGEMENT: EVALUATE EXISTING POSITIONS FOR SELLING
        # ====================================================================
        # Check each current holding against exit criteria
        for ticker, portfolio_weight in current_portfolio_weights.items():
            self.sell(date, ticker, tickers_df, complement_df, portfolio_weight)

        # ====================================================================
        # ENTRY MANAGEMENT: EVALUATE ALL TICKERS FOR BUYING
        # ====================================================================
        # Score all potential investments (existing and new)
        tickers_score = self.score_tickers(date, tickers_df, complement_df)

        # Execute buy decisions based on scores
        self.buy(date, tickers_score, daily_ticker_data, daily_index_data)

        # ====================================================================
        # CASH MANAGEMENT: REINVEST REMAINING CASH
        # ====================================================================
        # Put any remaining cash back into the default index
        # This ensures we're always fully invested
        self.portfolio.buy_default_index_with_all_cash(current_index_price, date)

    def trade(self, tickers_df, complement_df, default_index, outputpath=None,
              start_date=None, end_date=None):
        """
        Execute the complete trading simulation over a specified date range.

        This method runs the entire backtesting simulation, calling trade_recurrent
        for each trading day in the specified period.

        Args:
            tickers_df: Complete historical price and technical data
            complement_df: Complete analyst recommendation data
            default_index: Complete default index data
            outputpath: Optional path to save trading results
            start_date: Optional start date for simulation
            end_date: Optional end date for simulation
        """
        # ====================================================================
        # SIMULATION INITIALIZATION
        # ====================================================================
        # Start with fresh portfolio
        self.portfolio = Portfolio()

        # Get list of all tradeable tickers
        tickers =  list(set(complement_df.ticker).intersection(set(tickers_df.ticker)))

        # ====================================================================
        # DATE RANGE SETUP
        # ====================================================================
        # Get all available trading dates
        trading_dates = tickers_df.Date
        trading_dates = np.array(sorted(list(set(trading_dates))))

        # Apply date range filters if specified
        if start_date is not None:
            trading_dates = trading_dates[
                trading_dates >= pd.Timestamp(start_date).tz_localize('UTC')
                ]
        if end_date is not None:
            trading_dates = trading_dates[
                trading_dates <= pd.Timestamp(end_date).tz_localize('UTC')
                ]

        # ====================================================================
        # SIMULATION EXECUTION
        # ====================================================================
        # Execute trading strategy for each day in the simulation period
        for date in trading_dates:
            self.trade_recurrent(date, tickers, tickers_df, complement_df, default_index)

        # ====================================================================
        # RESULTS EXPORT
        # ====================================================================
        # Save trading history if output path is specified
        if outputpath is not None:
            os.makedirs(outputpath, exist_ok=True)
            output_file = os.path.join(outputpath, 'trade_simulation_results.csv')
            self.portfolio.history_to_csv(output_file)
            print(f"Trading simulation results saved to: {output_file}")


if __name__ == "__main__":

    policy = TradingPolicy.create("MostBasic", config= ConfigManager())
    print(policy.name)

