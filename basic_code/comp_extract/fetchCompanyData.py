"""
Company Data Fetcher Module

This module handles the retrieval and processing of company data from various
financial data sources. It provides functionality to fetch and organize financial
statements, market data, and company information.

Key Features:
- Financial data retrieval
- Company information gathering
- Data validation and cleaning
- Multiple data source integration
- Data caching and storage

Author: talhadaski
Last updated: 2024-03-29
"""

import requests
import yfinance as yf
from yahoofinancials import YahooFinancials
import json
import pandas as pd
import os
import requests
import earningscall
from earningscall import get_company
from alphavantage_api_client import AlphavantageClient, GlobalQuote, AccountingReport
import alpha_vantage
from yahoo_fin.stock_info import get_data, get_income_statement,tickers_sp500, tickers_nasdaq, tickers_other, get_quote_table
import time
from polygon import RESTClient
import pickle


class FinancialDataDownloader:
    def __init__(self, ticker_file, results_path, api_key_earning,api_key_fundamental,api_key_earningcall, years, quarters):
        self.ticker_file = ticker_file
        self.results_path = results_path
        self.api_key_earning = api_key_earning
        self.api_key_fundamental = api_key_fundamental
        self.years = years
        self.quarters = quarters
        earningscall.api_key = api_key_earningcall

    def load_tickers(self):
        with open(self.ticker_file, 'r') as file:
            return json.load(file)

    def write_to_file(self, ticker, x, i, text):
        filename = f"{self.results_path}earningCalls/{ticker}_{x}_{i}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            file.write(text)

    def download_earnings_calls(self, ticker):

        for year in years:
            for quarter in quarters:
                api_url = f'https://api.api-ninjas.com/v1/earningstranscript?ticker={ticker}&year={year}&quarter={quarter}'
                response = requests.get(api_url, headers={'X-Api-Key': self.api_key_earnings})
                if response.status_code == requests.codes.ok:
                    self.write_to_file(ticker, year, quarter, response.text)
                else:
                    print(f"Error: {response.status_code}, {response.text}")



    def download_all_data(self):
        tickers = self.load_tickers()
        client = AlphavantageClient().with_api_key(self.api_key_fundamental)

        for ticker in tickers:
            print(f"Processing {ticker}")
            #self.download_earnings_calls(ticker)
            self.download_financial_numbers(ticker)
            #self.download_historical_data(ticker)




    def get_historical_eps(self,ticker, start_date, end_date):
        yahoo_financials = YahooFinancials(ticker)
        financials_data = yahoo_financials.get_historical_earnings(ticker, start_date, end_date)

        # Extracting EPS data from the financials data
        eps_data = [
            {
                'date': record['startdatetime'],
                'eps': record['epsactual']
            }
            for record in financials_data['earningsHistory']['history']
        ]

        # Convert to DataFrame for easier manipulation
        eps_df = pd.DataFrame(eps_data)
        return eps_df

    # Function to find the closest date
    def find_closest_date(self,date, date_list):
        return min(date_list, key=lambda x: abs(x - date))

    def download_earningcall(self, ticker):
        #earningscall.api_key = "NtcGbV1W52UshmWU0Ag3fQ"
        if not os.path.exists(self.results_path + ticker):
            os.makedirs(self.results_path+ticker )
        company = earningscall.get_company(ticker)  # Lookup Apple, Inc by its ticker symbol, "AAPL"
        if company is None:
            return
        for year in self.years:
            for quarter in self.quarters:
                transcript = company.get_transcript(year=year, quarter=quarter,level = 2)

                if transcript is not None:
                    mySpeakers = []
                    speakers = transcript.speakers
                    if transcript.event is not None and transcript.event.conference_date is not None:
                        mySpeakers.append(dict(date = (transcript.event.conference_date).isoformat()))
                    for i in range(len(speakers)):
                        if speakers[i].speaker_info is not None:
                            mySpeaker = dict(speaker_name=speakers[i].speaker_info.name,speaker_title=speakers[i].speaker_info.title, text=speakers[i].text)
                        else:
                            mySpeaker = dict(speaker_name=None,
                                             speaker_title=None, text=speakers[i].text)
                        mySpeakers.append(mySpeaker)

                    json_data = json.dumps(mySpeakers, indent=4)

                    with open(self.results_path+ticker+'/parsedEarning_'+str(year)+'_'+str(quarter)+'.json', "w") as json_file:
                        json_file.write(json_data)
                    # Open the file in write mode and write the string to it
                    with open(self.results_path+ticker+'/earning_'+str(year)+'_'+str(quarter)+'.json', "w", encoding="utf-8") as file:
                        file.write(transcript.text)
                    aaa = 5
        aaa = 5

    def download_financial_numbers(self,ticker):
        import os
        if not os.path.exists(self.results_path + ticker):
            os.makedirs(self.results_path+ticker )

        # calculate market capitalization
        client = AlphavantageClient().with_api_key(self.api_key_fundamental)
        earnings = json.loads(client.get_earnings(ticker).model_dump_json())
        with open(self.results_path + ticker+'\\earnings.json', "w", encoding='utf-8') as json_file:
            json_file.write(client.get_earnings(ticker).model_dump_json())
        cash_flow = json.loads(client.get_cash_flow(ticker).model_dump_json())
        with open(self.results_path + ticker + '\\cash_flow.json', "w", encoding='utf-8') as json_file:
            json_file.write(client.get_cash_flow(ticker).model_dump_json())
        balance_sheet = json.loads(client.get_balance_sheet(ticker).model_dump_json())
        with open(self.results_path + ticker + '\\balance_sheet.json', "w", encoding='utf-8') as json_file:
            json_file.write(client.get_balance_sheet(ticker).model_dump_json())
        income_statement = json.loads(client.get_income_statement(ticker).model_dump_json())
        with open(self.results_path + ticker + '\\income_statement.json', "w", encoding='utf-8') as json_file:
            json_file.write(client.get_income_statement(ticker).model_dump_json())
        company_overview = json.loads(client.get_company_overview(ticker).model_dump_json())
        with open(self.results_path + ticker + '\\company_overview.json', "w", encoding='utf-8') as json_file:
            json_file.write(client.get_company_overview(ticker).model_dump_json())
        stockPrice = json.loads(client.get_daily_quote(ticker).model_dump_json())
        #quote = json.loads(client.get_technical_indicator(ticker).model_dump_json())
        #with open(self.results_path + ticker+'\\quote.json', "w") as json_file:
        #    json_file.write(client.get_technical_indicator(ticker).model_dump_json())
        # client.get_earnings_calendar(ticker)

        # get stock price data
        df_prices = pd.DataFrame(stockPrice['data']).T
        df_prices.index = pd.to_datetime(df_prices.index)
        df_prices = df_prices.astype(float)
        # Convert the index to datetime
        df_prices.index = pd.to_datetime(df_prices.index)
        # Filter the DataFrame to keep only the rows starting from 2020-01-01
        filtered_df = df_prices[df_prices.index >= '2019-01-01']
        filtered_df.to_excel(self.results_path + ticker+'\\stockPrice.xlsx', index=True, index_label='Date')

        # get fundamental data
        united_dict = [{**earnings['quarterlyReports'][i], **income_statement['quarterlyReports'][i], **balance_sheet['quarterlyReports'][i]} for i in range(min(len(earnings['quarterlyReports']),len(income_statement['quarterlyReports']),len(balance_sheet['quarterlyReports'])))]
        # Fields to keep
        fields_to_keep = ["fiscalDateEnding","reportedDate","totalRevenue","reportedEPS","estimatedEPS", "grossProfit", "totalRevenue", "operatingIncome", "researchAndDevelopment","incomeBeforeTax","ebit", "ebitda", "netIncome", "commonStockSharesOutstanding","totalAssets","totalLiabilities","totalShareholderEquity"]
        # Create a new list of dictionaries with only the desired fields
        subset_list_of_dicts = [
            {field: item[field] for field in fields_to_keep} for item in united_dict
        ]

        # Convert reportedDate strings to datetime objects and numerical values in financial data
        for data in subset_list_of_dicts:
            data['reportedDate'] = pd.to_datetime(data['reportedDate'])
            #data['totalRevenue'] = float(data['totalRevenue'])
            #data['netIncome'] = float(data['netIncome'])



            # Find the closest stock price date for the reported date
            closest_date = self.find_closest_date(data['reportedDate'], df_prices.index)
            stock_price = df_prices.loc[closest_date, '4. close']
            if (data['commonStockSharesOutstanding'] is not None) and (data['commonStockSharesOutstanding'] != 'None'):
                data['commonStockSharesOutstanding'] = float(data['commonStockSharesOutstanding'])
                data['MarketCap'] = stock_price * data['commonStockSharesOutstanding']
            else:
                data['MarketCap'] = None
            # Calculate market cap

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(subset_list_of_dicts)
        df.to_excel(self.results_path + ticker + '\\financial_data.xlsx', index=False)

    @staticmethod
    def getStocksList(filename):
        # Load data from an Excel file
        df = pd.read_csv(filename)

        # Convert ipoDate to datetime
        df['ipoDate'] = pd.to_datetime(df['ipoDate'])

        # Filter companies with IPO date earlier than June 24 of any year and leave only stocks
        #filtered_df = df[(df['ipoDate'].dt.month < 6) & (df['assetType'] == 'Stock')]
        filtered_df = df[df['assetType'] == 'Stock']
        stocksList = filtered_df['symbol'].to_list()
        # Display the filtered DataFrame
        return stocksList

    @staticmethod
    def getStocksList2(badStocks,goodStocks):
        # Load bad stocks from JSON file
        with open(badStocks, 'r') as file:
            bad_stocks = json.load(file)

        # Load good stocks from JSON file
        with open(goodStocks, 'r') as file:
            good_stocks = json.load(file)

        # Combine the ticker names into one list
        all_stocks = bad_stocks + good_stocks
        return all_stocks




if __name__ == "__main__":
    # Example paths and API key, replace with actual paths and keys
    ticker_file = 'good_stocks.json'
    data_path = '../data/'
    tickers_path = '../data/tickers/'
    api_key_earning = 'QkoFJYm092BLLvX8iluF3A==ju732rgb1V4WMLsQ'
    api_key_fundamental = 'Z97ZIJMOC7UKBNGI'
    api_key_earningCall = "NtcGbV1W52UshmWU0Ag3fQ"
    stockList = 'listing_status.csv'
    quarters = [1, 2, 3, 4]
    years = [ 2020,2021,2022, 2023, 2024]
    downloader = FinancialDataDownloader(ticker_file, tickers_path, api_key_earning, api_key_fundamental,
                                         api_key_earningCall,years,quarters)

    downloader.download_financial_numbers('SPY')
    stocksList = downloader.getStocksList(data_path+stockList)
    idx = stocksList.index('SPY')
    stocksList = stocksList[idx:]
    for ticker in stocksList:
        print(f"Processing {ticker}")
        # self.download_earnings_calls(ticker)
        downloader.download_financial_numbers(ticker)
        #downloader.download_earningcall(ticker)

        # self.download_historical_data(ticker)

