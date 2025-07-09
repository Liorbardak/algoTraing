import numpy as np
import pandas as pd
import os
import pylab as plt
from utils.visualizer import plot_ticker, plot_overall, holding_per_time
from utils.report_utils import HtmlReport


def tradesim_report(tickers_df, complement_df, snp_df, trade_hist_df, outputdir):
    """
    Generate a comprehensive trading simulation report comparing portfolio performance
    against S&P 500 benchmark.

    This function creates an HTML report with performance metrics, visualizations,
    and individual ticker analysis for a trading simulation.

    Args:
        tickers_df (pd.DataFrame): Historical price data for individual tickers
        complement_df (pd.DataFrame): Complementary data containing ticker metadata
        snp_df (pd.DataFrame): S&P 500 historical data for benchmarking
        trade_hist_df (pd.DataFrame): Trading history with portfolio values over time
        outputdir (str): Directory path where the HTML report will be saved

    Returns:
        None: Saves HTML report to specified output directory
    """

    # ========================================================================
    # DATA PREPROCESSING AND DATE ALIGNMENT
    # ========================================================================

    # Convert trade history dates to pandas Timestamp objects for consistency
    trade_hist_df.Date = [pd.Timestamp(v) for v in trade_hist_df.Date.values]

    # Determine overlapping date range between trading history and S&P 500 data
    # This ensures we're comparing apples to apples in our analysis
    start_date = np.max([trade_hist_df.Date.min(), snp_df.Date.min()])
    end_date = np.min([trade_hist_df.Date.max(), snp_df.Date.max()])

    # Filter all dataframes to the common date range
    snp_df = snp_df[
        (snp_df.Date >= start_date) & (snp_df.Date <= end_date)
        ]
    trade_hist_df = trade_hist_df[
        (trade_hist_df.Date >= start_date) & (trade_hist_df.Date <= end_date)
        ]
    tickers_df = tickers_df[
        (tickers_df.Date >= start_date) & (tickers_df.Date <= end_date)
        ]

    # ========================================================================
    # TICKER ANALYSIS AND CATEGORIZATION
    # ========================================================================

    # Extract all unique tickers from the complement dataset
    all_tickers = list(set(complement_df.ticker))

    # Identify which tickers were actually traded (appear in trade history columns)
    tickers_that_were_in_portfolio = [
        ticker for ticker in trade_hist_df.keys()
        if ticker in all_tickers
    ]

    # Identify tickers that were available but never traded
    tickers_that_were_not_in_portfolio = (
            set(all_tickers) - set(tickers_that_were_in_portfolio)
    )

    # ========================================================================
    # YEARLY PERFORMANCE CALCULATION
    # ========================================================================

    # Calculate year-by-year performance metrics
    start_of_year = start_date
    performance_results = []

    while start_of_year <= end_date:
        # Define end of current year
        end_of_year = start_of_year.replace(month=12, day=31)

        # Calculate portfolio profit for the year
        year_start_value = trade_hist_df[
            trade_hist_df.Date >= start_of_year
            ].total_value.values[0]
        year_end_value = trade_hist_df[
            trade_hist_df.Date <= end_of_year
            ].total_value.values[-1]
        portfolio_profit = (year_end_value / year_start_value) - 1

        # Calculate S&P 500 profit for the same period
        snp_start_value = snp_df[
            snp_df.Date >= start_of_year
            ].Close.values[0]
        snp_end_value = snp_df[
            snp_df.Date <= end_of_year
            ].Close.values[-1]
        snp_profit = (snp_end_value / snp_start_value) - 1

        # Store results as percentage with 1 decimal place
        performance_results.append({
            'year': start_of_year.year,
            'profit[%]': np.round(portfolio_profit * 100, 1),
            'snp_profit[%]': np.round(snp_profit * 100, 1)
        })

        # Move to next year
        start_of_year = start_of_year.replace(
            year=start_of_year.year + 1,
            month=1, day=1, hour=0, minute=0, second=0, microsecond=0
        )

    # ========================================================================
    # OVERALL PERFORMANCE CALCULATION
    # ========================================================================

    # Calculate total return over the entire period
    total_start_value = trade_hist_df[
        trade_hist_df.Date == start_date
        ].total_value.values[0]
    total_end_value = trade_hist_df[
        trade_hist_df.Date == end_date
        ].total_value.values[0]
    overall_portfolio_profit = (total_end_value / total_start_value) - 1

    # Calculate S&P 500 total return for comparison
    snp_total_start = snp_df[
        snp_df.Date == start_date
        ].Close.values[0]
    snp_total_end = snp_df[
        snp_df.Date == end_date
        ].Close.values[0]
    overall_snp_profit = (snp_total_end / snp_total_start) - 1

    # Add overall performance to results
    performance_results.append({
        'year': 'all',
        'profit[%]': np.round(overall_portfolio_profit * 100, 1),
        'snp_profit[%]': np.round(overall_snp_profit * 100, 1)
    })

    # ========================================================================
    # HTML REPORT GENERATION
    # ========================================================================

    # Initialize HTML report generator
    report = HtmlReport()

    # Add performance summary table
    report.add_df('Performance Results', pd.DataFrame(performance_results))

    # Add overall performance comparison chart (Portfolio vs S&P 500)
    overall_performance_fig = plot_overall(snp_df, trade_hist_df)
    report.add_figure('Portfolio vs S&P 500 Performance', overall_performance_fig)

    # Add portfolio holdings distribution over time
    holdings_fig = holding_per_time(trade_hist_df, tickers_that_were_in_portfolio)
    report.add_figure("Portfolio Holdings Over Time", holdings_fig)

    # ========================================================================
    # INDIVIDUAL TICKER ANALYSIS
    # ========================================================================

    # Generate detailed charts for each ticker that was traded
    # Sort tickers alphabetically for consistent report organization
    for ticker in sorted(tickers_that_were_in_portfolio):
        # Filter data for current ticker
        ticker_price_data = tickers_df[tickers_df.ticker == ticker]
        ticker_complement_data = complement_df[complement_df.ticker == ticker]

        # Generate ticker-specific visualization
        ticker_fig = plot_ticker(
            ticker,
            ticker_price_data,
            ticker_complement_data,
            trade_hist_df
        )

        # Add to report with ticker symbol as title
        report.add_figure(f"{ticker} Analysis", ticker_fig)

        # Clean up matplotlib figures to prevent memory issues
        plt.close("all")

    # ========================================================================
    # REPORT OUTPUT
    # ========================================================================

    # Save the complete HTML report to the specified output directory
    report_path = os.path.join(outputdir, 'trading_simulation_report.html')
    report.to_file(report_path)

    print(f"Trading simulation report successfully generated: {report_path}")