"""
Trading Simulator Module

This module provides a comprehensive trading simulation framework that allows users to:
- Test various trading policies on historical data
- Load and preprocess financial data from multiple sources
- Generate detailed performance reports and visualizations
- Configure simulation parameters through a centralized config system

Author: dadabardak
Last updated: 2024-03-29
Version: 1.0
"""
import pandas as pd
import os
from typing import Optional, Tuple
from datetime import datetime

from config.config import ConfigManager
from trade_policy import TradingPolicy
from data_loader import FinancialDataLoaderBase
from metrics import tradesim_report


class TrainingSimulator:
    """
    Main orchestrator for trading simulation and backtesting.

    This class coordinates all components of the trading simulation pipeline:
    - Data loading and preprocessing from multiple sources
    - Trading policy initialization and execution
    - Performance metrics calculation and reporting
    - Results storage and visualization

    The simulator supports flexible date ranges, multiple trading policies,
    and comprehensive output generation for analysis.

    Attributes:
        data_loader (FinancialDataLoaderBase): Handles all data loading operations
        trade_policy (TradingPolicy): Implements the trading strategy logic
        config (ConfigManager): Manages simulation configuration parameters
    """

    def __init__(self, config: ConfigManager) -> None:
        """
        Initialize the trading simulator with configuration.

        Args:
            config (ConfigManager): Configuration object containing all simulation parameters
                                  including data sources, policy settings, and output preferences
        """
        # Store configuration for reference
        self.config = config

        # Initialize data loader with configuration
        self.data_loader = FinancialDataLoaderBase(config)

        # Create trading policy based on configuration
        # Uses factory pattern to instantiate the correct policy type
        policy_name = config.get_parameter("policy", "policy_name")
        self.trade_policy = TradingPolicy.create(
            policy_name,
            config,
            default_index_name='snp'
        )

    def run_training_simulation(
            self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            outputpath: Optional[str] = None
    ) -> None:
        """
        Execute the complete trading simulation pipeline.

        This method orchestrates the entire simulation process:
        1. Load all required financial data (S&P 500, individual stocks, complements)
        2. Execute the trading policy over the specified date range
        3. Generate comprehensive performance reports and visualizations
        4. Save results to the specified output directory

        Args:
            start_date (Optional[str]): Start date for simulation in 'YYYY-MM-DD' format.
                                      If None, uses earliest available data date.
            end_date (Optional[str]): End date for simulation in 'YYYY-MM-DD' format.
                                    If None, uses latest available data date.
            outputpath (Optional[str]): Directory path for saving simulation results.
                                      If None, results are not saved to disk.

        Raises:
            FileNotFoundError: If required data files are not found
            ValueError: If date range is invalid or data is insufficient
        """
        # ==========================================
        # STEP 1: Load and prepare all data sources
        # ==========================================
        print("Loading S&P 500 index data...")
        snp_df = self.data_loader.load_snp()

        print("Loading individual stock data and complements...")
        tickers_df, complement_df, actual_min_max_dates, avg_df = self.data_loader.load_all_data()

        # Log data availability for debugging
        print(f"Data available from {actual_min_max_dates[0]} to {actual_min_max_dates[1]}")
        print(f"Loaded {len(tickers_df.columns)} individual stocks")
        print(f"Loaded {len(complement_df.columns)} complement instruments")

        # ==========================================
        # STEP 2: Execute trading policy
        # ==========================================
        print(f"Running trading simulation from {start_date} to {end_date}...")
        self.trade_policy.trade(
            tickers_df=tickers_df,
            complement_df=complement_df,
            default_index=snp_df,
            start_date=start_date,
            end_date=end_date,
            outputpath=outputpath
        )

        # ==========================================
        # STEP 3: Generate reports and analysis
        # ==========================================
        if outputpath is not None:
            print("Generating performance reports...")

            # Load trade history from the output file
            trade_history_file = os.path.join(outputpath, 'trade_sim.csv')

            if os.path.exists(trade_history_file):
                trade_hist_df = pd.read_csv(trade_history_file)

                # Generate comprehensive performance report
                tradesim_report(
                    tickers_df=tickers_df,
                    complement_df=complement_df,
                    snp_df=snp_df,
                    trade_hist_df=trade_hist_df,
                    outputdir=outputpath
                )

                print(f"Simulation completed successfully. Results saved to: {outputpath}")
            else:
                print(f"Warning: Trade history file not found at {trade_history_file}")
        else:
            print("Simulation completed. No output path specified - results not saved.")


def main(
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        outputpath: Optional[str] = None
) -> None:
    """
    Main entry point for the trading simulation.

    This function provides a clean interface for running simulations with
    minimal setup. It handles configuration loading and simulator initialization.

    Args:
        start_date (Optional[str]): Simulation start date in 'YYYY-MM-DD' format
        end_date (Optional[str]): Simulation end date in 'YYYY-MM-DD' format
        outputpath (Optional[str]): Directory path for saving results

    Example:
        >>> main(
        ...     start_date='2020-01-01',
        ...     end_date='2025-01-01',
        ...     outputpath='./results/my_simulation'
        ... )
    """
    # Initialize configuration manager
    config = ConfigManager()

    # Create and run trading simulator
    trading_simulator = TrainingSimulator(config=config)
    trading_simulator.run_training_simulation(
        start_date=start_date,
        end_date=end_date,
        outputpath=outputpath
    )


if __name__ == "__main__":
    # ==========================================
    # SIMULATION CONFIGURATION
    # ==========================================

    # Define simulation parameters
    START_DATE = '2020-01-01'  # Start of simulation period
    END_DATE = '2025-01-01'  # End of simulation period

    # Output directory for results (reports, charts, trade history)

    OUTPUT_PATH = 'C:/Users/dadab/projects/algotrading/results/trading_sim/test5'

    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # ==========================================
    # EXECUTE SIMULATION
    # ==========================================

    print("=" * 60)
    print("TRADING SIMULATION STARTING")
    print("=" * 60)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)

    # Run the simulation
    main(
        start_date=START_DATE,
        end_date=END_DATE,
        outputpath=OUTPUT_PATH
    )

    print("=" * 60)
    print("SIMULATION COMPLETED")
    print("=" * 60)
