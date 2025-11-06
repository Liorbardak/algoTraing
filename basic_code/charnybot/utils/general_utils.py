import os
import numpy as np
import json
import pandas as pd
import re
import shutil
import numpy as np

def compliment_format_converter(inpath , outpth):
    os.makedirs(outpth, exist_ok=True)
    tickers = [re.match(r'^([A-Z]+)', file).group(1) for file in os.listdir(inpath)]

    for ticker in os.listdir(inpath):
        if os.path.isdir(os.path.join(inpath, ticker)) == False:
            continue
        summery_compfile = os.path.join(inpath, ticker, 'summarizeCompliments_gpt41_2023.json')
        if os.path.isfile(summery_compfile):
            shutil.copy(summery_compfile, os.path.join(outpth, f'{ticker}_compliment_summary.json'))


def calculate_average_ticker(tickers_df):

    # Calculate the average of all stocks

    avg_df = None
    keys_to_avg = ['High', 'Low', 'Open', 'Close', 'AdjClose', 'Volume']

    # Calculate the average of all stocks
    for tdf in tickers_df:
        # Get the stock in the time range
        tdf = tdf[(pd.to_datetime(tdf.Date) >= actual_min_max_dates[0]) & (
                    pd.to_datetime(tdf.Date) <= actual_min_max_dates[1])]
        # Normalize price by the first date
        for k in keys_to_avg:
            tdf[k] = tdf[k] / tdf.Close.values[0]
        if avg_df is None:
            avg_df = tdf
        else:
            for k in keys_to_avg:
                avg_df[k] = avg_df[k].values + tdf[k].values

    for k in keys_to_avg:
        avg_df[k] = avg_df[k].values / len(tickers_df)


def weighted_adjustment(values, weights, adjustment):
    """
    Adjust sum of values by a certain amount, proportional to weights.

    Parameters:
    - values: array of positive numbers
    - weights: array of weights (higher weight = more change)
    - adjustment: amount to change total sum by (positive = increase, negative = decrease)

    Returns:
    - adjusted values (all positive)
    """
    values = np.array(values, dtype=float)
    weights = np.array(weights, dtype=float)
    # Normalize weights to sum to 1
    normalized_weights = weights / weights.sum()

    # Calculate adjustment for each element
    adjustments = normalized_weights * adjustment

    # Apply adjustments
    new_values = values + adjustments

    # For decreases, handle potential negative values
    if adjustment < 0:
        iteration = 0
        max_iterations = 100

        while (new_values < 0).any() and iteration < max_iterations:
            # Find negative values
            negative_mask = new_values < 0
            shortfall = -new_values[negative_mask].sum()

            # Set negative values to small positive
            new_values[negative_mask] = np.minimum(1e-3,values[negative_mask])

            # Redistribute shortfall among positive elements
            positive_mask = ~negative_mask
            if not positive_mask.any():
                break

            # Redistribute proportionally to weights of remaining elements
            remaining_weights = weights[positive_mask]
            remaining_weights_norm = remaining_weights / remaining_weights.sum()

            additional_reduction = remaining_weights_norm * shortfall
            new_values[positive_mask] -= additional_reduction

            iteration += 1

    return new_values


def weighted_increase(values, weights, increase_amount):
    """
    Increase sum by a certain amount, proportional to weights.
    Higher weight = more increase.
    """
    return weighted_adjustment(values, weights, increase_amount)


def weighted_decrease(values, weights, decrease_amount):
    """
    Decrease sum by a certain amount, proportional to weights.
    Higher weight = more decrease.
    """
    return weighted_adjustment(values, weights, -decrease_amount)



if __name__ == "__main__":
    # Example usage
    values = np.array([100, 50, 30, 20])
    weights = np.array([0.01, 0.1, 0.1, 0.9])
    reduction = -80

    result = weighted_adjustment(values, weights, reduction)
    print(f"Original: {values}, sum = {values.sum()}")
    print(f"Reduced: {result}, sum = {result.sum()}")
    print(f"Actual reduction: {values.sum() - result.sum()}")


    # inpath = 'C:/Users/dadab/projects/algotrading/data/gpt41_pretest'
    # outpth = 'C:/Users/dadab/projects/algotrading/data/complements/gpt41_2023'
    #
    # compliment_format_converter(inpath, outpth)
