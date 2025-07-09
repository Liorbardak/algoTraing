"""
Earnings Call Compliment Extraction Module

This module processes earnings call transcripts to extract and analyze analyst compliments
to company executives. It uses natural language processing and sentiment analysis to
identify and quantify positive feedback during earnings calls.

Key Features:
- Processes earnings call transcripts
- Identifies analyst compliments and their intensity
- Validates compliments against original text
- Aggregates compliment statistics
- Generates sentiment scores

Author: talhadaski
Last updated: 2024-03-29
"""

import os
import re
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from fetchCompanyData import FinancialDataDownloader
from activateChatty import ActivateChatty
from datetime import datetime
import shutil


class EarningsComplimentAnalyzer:
    """
    Analyzes earnings call transcripts to detect and validate analyst compliments.

    This class processes earnings call transcripts, identifies potential compliments,
    validates them, and aggregates the results for sentiment analysis.
    """

    def __init__(self, data_path, tickers_path, results_path):
        """
        Initialize the analyzer with paths to relevant data directories.

        Args:
            data_path (str): Path to the main data directory
            tickers_path (str): Path to the directory containing ticker data
            results_path (str): Path to store analysis results
        """
        self.data_path = data_path
        self.tickers_path = tickers_path
        self.results_path = results_path

    def is_json(self, text_string):
        """
        Check if a string is valid JSON.

        Args:
            text_string (str): String to check

        Returns:
            bool: True if the string is valid JSON, False otherwise
        """
        try:
            json.loads(text_string)
            return True
        except ValueError:
            return False

    def validate_compliments(self, compliments, activateChatty, prompt_validation_path, text, model="gpt-4o"):
        """
        Validate extracted compliments against the original text.

        For each compliment with a positive sentiment level, this method:
        1. Checks if the compliment actually appears in the text
        2. Double-checks using an LLM to confirm it's truly a compliment
        3. Updates the compliment's level based on validation results

        Args:
            compliments (list): List of compliment dictionaries
            activateChatty (ActivateChatty): API wrapper for LLM interactions
            prompt_validation_path (str): Path to the validation prompt template
            text (str): Original earnings call text
            model (str): LLM model to use for validation

        Returns:
            list: Updated list of validated compliments
        """
        for compliment in compliments:
            if int(compliment['level']) > 0:
                # Check if compliment appears in text
                compliment['appears_in_text'] = 1 if self.check_compliment_in_text(compliment, text) else 0

                if compliment['appears_in_text'] == 1:
                    # Double-check with LLM if it's truly a compliment
                    compliment['doublecheck_grade'] = activateChatty.activate(
                        prompt_validation_path, compliment['quoted_compliment'], "gpt-4o")

                    # Downgrade level if validation fails
                    if compliment['doublecheck_grade'] == 0:
                        compliment['level'] = 0
                        compliment['quoted_compliment'] = ''
                else:
                    # Not found in text, so not a valid compliment
                    compliment['level'] = 0
                    compliment['quoted_compliment'] = ''
            else:
                # Already marked as not a compliment
                compliment['level'] = 0
                compliment['quoted_compliment'] = ''

            # Clean up temporary validation fields
            compliment.pop('appears_in_text', None)
            compliment.pop('doublecheck_grade', None)

        # Remove duplicate compliments from the same analyst
        return self.remove_duplications(compliments)

    def find_substring_locations(self, source_text, substring):
        """
        Find all occurrences of a substring within a larger string.

        Args:
            source_text (str): The text to search in
            substring (str): The substring to find

        Returns:
            list: List of starting indices of all occurrences
        """
        return [i for i in range(len(source_text)) if source_text.startswith(substring, i)]

    def read_from_file(self, ticker, year, quarter):
        """
        Read earnings call text from a file.

        Args:
            ticker (str): Stock ticker symbol
            year (int): Year of the earnings call
            quarter (int): Quarter of the earnings call

        Returns:
            str: Content of the earnings call file
        """
        filename = f"earningCalls/{ticker}_{year}_{quarter}.txt"
        with open(filename, "r", encoding="utf-8") as file:
            content = file.read()
        return content

    def max_difference_index(self, numbers):
        """
        Find the index where the difference between consecutive values is maximum.

        Args:
            numbers (list): List of numerical values

        Returns:
            int: Index of maximum difference
        """
        return max(range(1, len(numbers)), key=lambda i: abs(numbers[i] - numbers[i - 1]))

    def split_text(self, text):
        """
        Split the earnings call text into segments around the Q&A section.

        Args:
            text (str): Full earnings call text

        Returns:
            list: List of text segments
        """
        # Find the Q&A section by locating the operator instructions
        idx = self.find_substring_locations(text, '[Operator Instructions]')
        if len(idx) > 1 and idx[0] < 10000:
            idx = idx[1]  # Use the second occurrence if the first is in the prepared remarks
        else:
            idx = idx[0]  # Use the first occurrence

        # Extract the Q&A section
        q_and_a_section = text[idx:]

        # Find all operator interventions to split into segments
        indices = self.find_substring_locations(q_and_a_section, 'Operator')

        # Add start and end indices
        indices = [0] + indices + [len(q_and_a_section)]

        # Split the text using the indices
        return [q_and_a_section[indices[i]:indices[i + 1]] for i in range(len(indices) - 1)]

    def normalize_text(self, text):
        """
        Normalize text by removing spaces, punctuation, and converting to lowercase.

        Args:
            text (str): Text to normalize

        Returns:
            str: Normalized text string
        """
        # Decode Unicode escape sequences
        normalized_text = text.encode().decode('unicode_escape')

        # Remove spaces, periods, and commas
        normalized_text = normalized_text.replace(" ", "")
        normalized_text = normalized_text.replace(".", "")
        normalized_text = normalized_text.replace(",", "")

        # Remove additional punctuation and whitespace, convert to lowercase
        normalized_text = re.sub(r'[.,] ', '', normalized_text).strip().lower()
        return normalized_text

    def check_compliment_in_text(self, compliment, text):
        """
        Check if a compliment appears in the text and meets validation criteria.

        Args:
            compliment (dict): Compliment dictionary
            text (str): Text to search in

        Returns:
            bool: True if compliment is found and valid, False otherwise
        """
        # Normalize the compliment text for comparison
        compliment_text = compliment['quoted_compliment'].lower()
        compliment_text = re.sub(r'[.,] ', '', compliment_text).strip()
        compliment_text = compliment_text.replace(" ", "")
        compliment_text = compliment_text.replace(".", "")
        compliment_text = compliment_text.replace(",", "")

        # Check if compliment is in the text
        is_good_compliment = False
        earning_text = self.normalize_text(text)

        # Look for a substantial portion of the compliment in the text (avoid very short matches)
        # Using a partial match to handle slight text variations
        if len(compliment_text) > 1 and compliment_text[1:min(len(compliment_text), 30)] in earning_text:
            # Filter out simple "thanks" or "great" which aren't substantial compliments
            words = compliment_text.split()
            if not (len(words) < 3 and (any(word.lower() == "thanks" for word in words) or
                                        any(word.lower() == "great" for word in words))):
                is_good_compliment = True

        return is_good_compliment

    def remove_duplications(self, compliments):
        """
        Remove duplicate compliments from the same analyst, keeping only the highest level.

        Args:
            compliments (list): List of compliment dictionaries

        Returns:
            list: Filtered list with duplicates removed
        """
        # Dictionary to hold the best compliment for each analyst
        analyst_compliments = {}

        # Find the highest level compliment for each analyst
        for compliment in compliments:
            analyst_name = compliment["analyst_name"]
            if (analyst_name not in analyst_compliments or
                    compliment["level"] > analyst_compliments[analyst_name]["level"]):
                analyst_compliments[analyst_name] = compliment

        # Mark duplicates in the original list
        for compliment in compliments:
            analyst_name = compliment["analyst_name"]
            best_compliment = analyst_compliments[analyst_name]["quoted_compliment"]
            compliment["is_duplication"] = 0 if (compliment["quoted_compliment"] == best_compliment) else 1

        # Filter out duplicates and remove the temporary field
        filtered_data = [item for item in compliments if item['is_duplication'] == 0]
        for item in filtered_data:
            item.pop("is_duplication", None)

        return filtered_data

    def store_clean_data(self, compliments_json):
        """
        Clean compliment data for storage by removing invalid entries and temporary fields.

        Args:
            compliments_json (list): List of compliment dictionaries

        Returns:
            list: Cleaned list of compliments
        """
        processed_data = []
        for item in compliments_json:
            # Skip duplicates
            if item.get('is_duplication', 0) == 1:
                continue

            # Clear invalid compliments
            if item.get('appears_in_text', 1) == 0 or item.get('doublecheck_grade', 1) == 0:
                item['quoted_compliment'] = ""
                item['level'] = 0

            # Remove temporary fields
            for field in ['appears_in_text', 'doublecheck_grade', 'is_duplication']:
                item.pop(field, None)

            processed_data.append(item)

        return processed_data

    def load_earnings_data(self, directory, ticker):
        """
        Load earnings call data and parsed results for a specific ticker.

        Args:
            directory (str): Directory containing the earnings data
            ticker (str): Stock ticker symbol

        Returns:
            list: List of dictionaries with earnings data and parsed results
        """
        earnings_list = []

        # Check if the directory exists
        if not os.path.exists(directory):
            print(f"No earnings data directory found for {ticker} at {directory}")
            return earnings_list

        for file_name in os.listdir(directory):
            # Only process earnings call files
            if file_name.startswith(('earning_', 'parsedEarning_')):
                # Extract file information
                try:
                    prefix, year, quarter = file_name.split('_')
                    quarter = int(quarter.split('.')[0])

                    # Read the file content
                    with open(os.path.join(directory, file_name), 'r', encoding='utf-8') as f:
                        if prefix == 'earning':
                            content = f.read()  # Read text content
                        elif prefix == 'parsedEarning':
                            content = json.load(f)  # Load JSON content

                    # Find or create entry for this year and quarter
                    entry = next((e for e in earnings_list if e['year'] == int(year) and
                                e['quarter'] == int(quarter)), None)

                    if not entry:
                        entry = {
                            'year': int(year),
                            'quarter': int(quarter),
                            'ticker': ticker,
                            'text': None,
                            'parsed_text': None
                        }
                        earnings_list.append(entry)

                    # Update the appropriate field
                    if prefix == 'earning':
                        entry['text'] = content
                    elif prefix == 'parsedEarning':
                        entry['parsed_text'] = content

                except Exception as e:
                    print(f"Error processing file {file_name} for {ticker}: {e}")
                    continue

        return earnings_list

    def aggregate_compliments(self, validated_compliments, date):
        """
        Aggregate compliment statistics for a single earnings call.

        Args:
            validated_compliments (list): List of validated compliment dictionaries
            date (str): Date of the earnings call

        Returns:
            dict: Dictionary of aggregated statistics
        """
        # Calculate summary statistics
        total_analysts = len(validated_compliments)
        analysts_level_1 = sum(1 for c in validated_compliments if c['level'] == 1)
        analysts_level_2 = sum(1 for c in validated_compliments if c['level'] == 2)
        analysts_level_3 = sum(1 for c in validated_compliments if c['level'] == 3)

        # Create a data dictionary for DataFrame construction
        data = {
            'date': date,
            'total_number_of_analysts': total_analysts,
            'number_of_analysts_comp_1': analysts_level_1,
            'number_of_analysts_comp_2': analysts_level_2,
            'number_of_analysts_comp_3': analysts_level_3
        }

        return data

    def check_ticker_already_processed(self, ticker):
        """
        Check if a ticker has already been processed by looking for summarizeCompliments file.

        Args:
            ticker (str): Stock ticker symbol

        Returns:
            bool: True if the ticker has already been processed, False otherwise
        """
        # Path to the summarized compliments file
        summary_file_path = os.path.join(self.results_path, ticker, 'summarizeCompliments.json')

        # Check if the file exists
        if os.path.exists(summary_file_path):
            try:
                # Verify it's a valid JSON file with content
                with open(summary_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # If the file contains data (not empty array or object)
                    if data and (isinstance(data, list) and len(data) > 0):
                        return True
            except:
                # If file exists but is invalid or empty, process it again
                return False

        return False


def get_sp500_tickers(path):
    """
    Load the list of S&P 500 stocks from a CSV file.

    Args:
        path (str): Path to the S&P 500 stocks file

    Returns:
        list: List of S&P 500 ticker symbols
    """
    try:
        # Try to load from CSV file (common format)
        if path.endswith('.csv'):
            df = pd.read_csv(path)
            # Check common column names for tickers
            if 'Symbol' in df.columns:
                return df['Symbol'].tolist()
            elif 'Ticker' in df.columns:
                return df['Ticker'].tolist()
            else:
                print(f"Could not find ticker column in {path}")
                return []
        # Try to load from JSON file
        elif path.endswith('.json'):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle different potential JSON structures
                if isinstance(data, list):
                    if isinstance(data[0], str):
                        return data
                    elif isinstance(data[0], dict) and ('symbol' in data[0] or 'ticker' in data[0]):
                        key = 'symbol' if 'symbol' in data[0] else 'ticker'
                        return [item[key] for item in data]
                elif isinstance(data, dict) and 'symbols' in data:
                    return data['symbols']
                elif isinstance(data, dict):
                    # Try to get values as tickers
                    return list(data.values()) if all(isinstance(v, str) for v in data.values()) else list(data.keys())
    except Exception as e:
        print(f"Error loading S&P 500 tickers from {path}: {e}")

    print(f"Could not load S&P 500 tickers from {path}, using backup method")
    # Fallback: Use a simple list of common S&P 500 companies as backup
    return ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'PG',
            'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'CRM', 'CMCSA', 'VZ', 'NFLX']


def main():
    """
    Main function to process earnings calls and extract analyst compliments.
    Modified with batch processing, timing measurements, and optimized output.
    """
    import time
    
    # Configuration paths
    basedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_path = os.path.join(basedir,'data')
    tickers_path = os.path.join(data_path,'tickers')

    
    # Create timestamp-based results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder_name = f"results_{timestamp}"
    #results_path = f'../../data/results/{results_folder_name}/'
    results_path = os.path.join(data_path, results_folder_name)
    prompt_path = '../../data/prompts/'

    # Prompt file paths
    prompt_detection_path = prompt_path + 'detection.txt'
    prompt_validation_path = prompt_path + 'validation.txt'

    # Initialize the analyzer classes
    analyzer = EarningsComplimentAnalyzer(data_path, tickers_path, results_path)
    chatty = ActivateChatty()

    # Process these tickers
    # target_tickers = ["ACAD", "ACB", "ACHC", "ADMA", "AES", "AKAM", "AMRC", "AMRN", "AMSC", "AMPX", "ANF", "AQST", "ASLE", "ASTH", "AVAH", "AVDX", "AVGO", "AVPT", "BKE", "BKTI", "BBCP", "BMA", "BNS", "BWAY", "BBAR", "BTBT", "CAPL", "CAR", "CCBS", "CCBG", "CERS", "CIFR", "CLS", "COIN", "CRDO", "CRS", "CSAN", "CSIQ", "DAKT", "DAY", "DIN", "DRD", "DSP", "DUOL", "EAT", "EIC", "EDN", "ESOA", "ETN", "ETSY", "EXFY", "EXK", "FBMS", "FGBI", "FICO", "FIVE", "FMC", "FMAO", "FNB", "FORR", "FRPH", "FSM", "FTK", "FTAI", "GHM", "GMAB", "GMED", "GNP", "GOSS", "GPN", "HEPS", "HBT", "HL", "HLF", "HNVR", "HOOD", "HRTG", "IART", "IESC", "IREN", "INMD", "IRWD", "KEQU", "KINS", "KOPN", "LFVN", "LMB", "LOCO", "LULU", "LWAY", "MAMA", "MARA", "MASI", "META", "MET", "MHK", "MLCO", "MMYT", "MOD", "MODG", "MOV", "MRAM", "MPTI", "MTRN", "MYO", "MYPS", "MNTK", "NBTA", "NBTB", "NFE", "NGVC", "NL", "NOVT", "NNI", "NRC", "NXT", "ODP", "OFLX", "OKTA", "OLP", "OPK", "OSCR", "OTRK", "OZK", "PAMT", "PBA", "PAY", "PCB", "PEBO", "PERF", "PLTR", "PM", "POOL", "PSFE", "PSIX", "PSX", "PUBM", "POWL", "PRU", "RCL", "RDWR", "REAX", "RGLD", "RKLB", "RRBI", "RRR", "SFL", "SGML", "SHEL", "SHYF", "SITC", "SITE", "SKYW", "SLQT", "SMBC", "SMBK", "SOL", "SPH", "SPOT", "SPTN", "SSTI", "STAA", "STBA", "STRL", "SU", "SUN", "SVV", "SWBI", "SYBT", "TALK", "TATT", "TBLA", "TGTX", "TOWN", "TRP", "TSEM", "TS", "TSCO", "TV", "TZOO", "UL", "UGP", "UTI", "UVSP", "VOYA", "VRT", "VST", "WB", "WBS", "WD", "WDH", "WOOF", "WULF", "XPO", "YPF", "ZI", "ZION"]

    # target_tickers = ["CYBR"]
    target_tickers = ["ADM","CLBT"]

    print(f"Processing {len(target_tickers)} tickers with batch optimization and timing measurements")
    print(f"Target tickers: {', '.join(target_tickers)}")
    print(f"Output directory: {results_path}")

    # Create results directory if it doesn't exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)
        
    # Copy prompt files to results directory for reference
    try:
        shutil.copy2(prompt_detection_path, os.path.join(results_path, 'detection_prompt_used.txt'))
        shutil.copy2(prompt_validation_path, os.path.join(results_path, 'validation_prompt_used.txt'))
        print(f"✓ Copied prompt files to results directory: {results_path}")
    except Exception as e:
        print(f"Warning: Could not copy prompt files: {e}")

    # Track overall progress
    overall_start_time = time.time()
    processed_count = 0
    failed_count = 0
    total_api_calls = 0

    # Process each ticker
    for i, ticker in enumerate(target_tickers, 1):
        print(f"\n{'='*80}")
        print(f"Processing ticker {i}/{len(target_tickers)}: {ticker.upper()}")
        print(f"{'='*80}")
        ticker_start_time = time.time()

        # Load earnings calls data
        earnings = analyzer.load_earnings_data(os.path.join(tickers_path, ticker), ticker)

        if not earnings:
            print(f"No earnings data found for {ticker}")
            failed_count += 1
            continue

        # Filter valid earnings calls
        valid_earnings = [call for call in earnings if call['text'] is not None and call['parsed_text'] is not None]
        print(f"Found {len(valid_earnings)} valid earnings calls")

        if not valid_earnings:
            print(f"No valid earnings data for {ticker}")
            failed_count += 1
            continue

        try:
            # STAGE 1: Individual Detection (gpt-4o-mini)
            print("\nSTAGE 1: Individual Detection Phase")
            print("-" * 50)
            stage1_start = time.time()
            all_quarter_compliments = {}
            stage1_api_time = 0
            stage1_processing_time = 0
            
            for call in valid_earnings:
                quarter_key = f"{call['year']}_Q{call['quarter']}"
                print(f"  Detecting compliments for {quarter_key}...")
                
                # API call timing
                api_start = time.time()
                tentative_compliments = chatty.activate(
                    prompt_detection_path,
                    call['parsed_text'],
                    "gpt-4.1-mini"
                )
                api_end = time.time()
                call_duration = api_end - api_start
                stage1_api_time += call_duration
                total_api_calls += 1
                print(f"    API call completed in {call_duration:.2f} seconds")
                
                # Processing timing
                process_start = time.time()
                
                # Safety check for API response
                if not isinstance(tentative_compliments, list):
                    print(f"    Warning: Unexpected API response format for {quarter_key}")
                    tentative_compliments = []
                
                # Get date from parsed text
                date = None
                if isinstance(call['parsed_text'], list) and len(call['parsed_text']) > 0:
                    date = call['parsed_text'][0].get('date') if isinstance(call['parsed_text'][0], dict) else None
                if date is None:
                    date = f"{call['year']}-{call['quarter']*3:02d}-01"
                
                # Store for batch validation
                all_quarter_compliments[quarter_key] = {
                    'year': call['year'],
                    'quarter': call['quarter'],
                    'date': date,
                    'compliments': tentative_compliments,
                    'original_text': call['text']  # Keep for validation
                }
                
                process_end = time.time()
                stage1_processing_time += (process_end - process_start)
                
                print(f"    Found {len([c for c in tentative_compliments if int(c.get('level', 0)) > 0])} potential compliments")
                
            stage1_end = time.time()
            stage1_total = stage1_end - stage1_start
            
            print(f"\nSTAGE 1 COMPLETED:")
            print(f"  Total time: {stage1_total:.2f} seconds")
            print(f"  API calls time: {stage1_api_time:.2f} seconds ({stage1_api_time/stage1_total*100:.1f}%)")
            print(f"  Processing time: {stage1_processing_time:.2f} seconds ({stage1_processing_time/stage1_total*100:.1f}%)")
            print(f"  Other operations: {stage1_total - stage1_api_time - stage1_processing_time:.2f} seconds")
            
            # Save detected compliments before validation
            print(f"\nSaving detected compliments before validation...")
            detection_only_filename = os.path.join(results_path, f'{ticker}_detected_compliments_before_validation.json')
            
            # Remove original_text for the save (not needed in output)
            detection_save_data = {}
            for quarter_key, quarter_data in all_quarter_compliments.items():
                detection_save_data[quarter_key] = {
                    'year': quarter_data['year'],
                    'quarter': quarter_data['quarter'],
                    'date': quarter_data['date'],
                    'compliments': quarter_data['compliments']
                }
            
            with open(detection_only_filename, 'w', encoding='utf-8') as f:
                json.dump(detection_save_data, f, ensure_ascii=False, indent=4)
            print(f"✓ Saved detected compliments: {detection_only_filename}")
            
            # STAGE 2: Batch Validation (gpt-4o)
            print(f"\nSTAGE 2: Batch Validation Phase")
            print("-" * 50)
            stage2_start = time.time()
            
            # Count potential compliments before batch validation
            potential_compliments_count = sum(
                len([c for c in qd['compliments'] if int(c.get('level', 0)) > 0]) 
                for qd in all_quarter_compliments.values()
            )
            print(f"  Found {potential_compliments_count} potential compliments to validate")
            
            stage2_api_time = 0
            stage2_processing_time = 0
            validation_successful = False
            
            if potential_compliments_count > 0:
                # Prepare batch validation input
                prep_start = time.time()
                batch_validation_input = prepare_batch_validation_input(all_quarter_compliments)
                prep_end = time.time()
                
                print(f"  Batch preparation completed in {prep_end - prep_start:.2f} seconds")
                
                # Convert dictionary to JSON string for the API call
                try:
                    batch_validation_input_str = json.dumps(batch_validation_input, ensure_ascii=False, indent=2)
                    
                    # Single API call for all verification
                    api_start = time.time()
                    batch_validated_response = chatty.activate(
                        prompt_validation_path,
                        batch_validation_input_str,
                        "gpt-4o"
                    )
                    api_end = time.time()
                    stage2_api_time = api_end - api_start
                    total_api_calls += 1
                    print(f"  Batch validation API call completed in {stage2_api_time:.2f} seconds")
                    
                    # Parse batch validation response
                    parse_start = time.time()
                    if isinstance(batch_validated_response, dict) and len(batch_validated_response) > 0:
                        validated_quarter_compliments = parse_batch_validation_response(
                            batch_validated_response, all_quarter_compliments
                        )
                        validation_successful = True
                        print(f"  ✅ Validation successful with {len(batch_validated_response)} quarters processed")
                    elif isinstance(batch_validated_response, list) and len(batch_validated_response) > 0:
                        validated_quarter_compliments = parse_batch_validation_response(
                            batch_validated_response, all_quarter_compliments
                        )
                        validation_successful = True
                        print(f"  ✅ Validation successful with {len(batch_validated_response)} validation results")
                    else:
                        validated_quarter_compliments = all_quarter_compliments
                        validation_successful = False
                        print(f"  ❌ Validation failed - JSON parsing error or empty response")
                    parse_end = time.time()
                    stage2_processing_time = (prep_end - prep_start) + (parse_end - parse_start)
                    
                    # Verify validation was applied
                    validation_applied_count = sum(
                        len([c for c in qd['compliments'] if int(c.get('level', 0)) > 0]) 
                        for qd in validated_quarter_compliments.values()
                    )
                    print(f"  Validation completed. Remaining valid compliments: {validation_applied_count}")
                    
                except Exception as e:
                    print(f"  ❌ Error in validation process: {e}")
                    validation_successful = False
                    validated_quarter_compliments = all_quarter_compliments
                    stage2_api_time = 0
                    stage2_processing_time = (prep_end - prep_start)
            else:
                print("  No compliments found to validate")
                validated_quarter_compliments = all_quarter_compliments
                validation_successful = True  # No validation needed, so consider it successful
            
            stage2_end = time.time()
            stage2_total = stage2_end - stage2_start
            
            print(f"\nSTAGE 2 COMPLETED:")
            print(f"  Total time: {stage2_total:.2f} seconds")
            print(f"  API calls time: {stage2_api_time:.2f} seconds ({stage2_api_time/stage2_total*100:.1f}%)")
            print(f"  Processing time: {stage2_processing_time:.2f} seconds ({stage2_processing_time/stage2_total*100:.1f}%)")
            
            # STAGE 2.5: Remove duplicate analysts (keeping only the highest level compliment per analyst)
            print(f"\nRemoving duplicate analysts...")
            dedup_start = time.time()
            
            for quarter_key, quarter_data in validated_quarter_compliments.items():
                original_count = len(quarter_data['compliments'])
                quarter_data['compliments'] = analyzer.remove_duplications(quarter_data['compliments'])
                dedup_count = len(quarter_data['compliments'])
                if original_count != dedup_count:
                    print(f"  {quarter_key}: Removed {original_count - dedup_count} duplicate analysts")
                    
            dedup_end = time.time()
            print(f"✓ Duplicate removal completed in {dedup_end - dedup_start:.2f} seconds")
            
            # STAGE 3: Generate Output Files (only if validation was successful)
            if validation_successful:
                print(f"\nSTAGE 3: Output Generation")
                print("-" * 50)
                stage3_start = time.time()
                
                # Remove original_text before saving (not needed in output)
                for quarter_data in validated_quarter_compliments.values():
                    quarter_data.pop('original_text', None)
                
                # File 1: All compliments from all quarters
                consolidated_filename = os.path.join(results_path, f'{ticker}_all_validated_compliments.json')
                with open(consolidated_filename, 'w', encoding='utf-8') as f:
                    json.dump(validated_quarter_compliments, f, ensure_ascii=False, indent=4)
                print(f"  ✓ Saved consolidated compliments: {consolidated_filename}")
                
                # File 2: Summary statistics
                aggregate_results = []
                total_analysts = 0
                total_valid_compliments = 0
                
                for quarter_key, quarter_data in validated_quarter_compliments.items():
                    compliments = quarter_data['compliments']
                    date = quarter_data['date']
                    
                    # Count analysts and compliments
                    quarter_analysts = len(compliments)
                    quarter_level_1 = sum(1 for c in compliments if int(c.get('level', 0)) == 1)
                    quarter_level_2 = sum(1 for c in compliments if int(c.get('level', 0)) == 2)
                    quarter_level_3 = sum(1 for c in compliments if int(c.get('level', 0)) == 3)
                    
                    total_analysts += quarter_analysts
                    total_valid_compliments += (quarter_level_1 + quarter_level_2 + quarter_level_3)
                    
                    summary_stats = {
                        'date': date,
                        'quarter': quarter_key,
                        'total_number_of_analysts': quarter_analysts,
                        'number_of_analysts_comp_1': quarter_level_1,
                        'number_of_analysts_comp_2': quarter_level_2,
                        'number_of_analysts_comp_3': quarter_level_3
                    }
                    aggregate_results.append(summary_stats)
                
                summary_filename = os.path.join(results_path, f'{ticker}_compliment_summary.json')
                with open(summary_filename, 'w', encoding='utf-8') as f:
                    json.dump(aggregate_results, f, ensure_ascii=False, indent=4)
                print(f"  ✓ Saved summary statistics: {summary_filename}")
                
                stage3_end = time.time()
                stage3_total = stage3_end - stage3_start
                
                print(f"\nSTAGE 3 COMPLETED:")
                print(f"  Total time: {stage3_total:.2f} seconds")
                
                # Ticker Summary
                ticker_end_time = time.time()
                ticker_total = ticker_end_time - ticker_start_time
                
                print(f"\n🎉 TICKER {ticker.upper()} COMPLETED:")
                print(f"  📊 Statistics:")
                print(f"    - Total quarters processed: {len(validated_quarter_compliments)}")
                print(f"    - Total analysts: {total_analysts}")
                print(f"    - Total valid compliments: {total_valid_compliments}")
                print(f"  ⏱️  Timing breakdown:")
                print(f"    - Stage 1 (Detection): {stage1_total:.2f}s ({stage1_total/ticker_total*100:.1f}%)")
                print(f"    - Stage 2 (Validation): {stage2_total:.2f}s ({stage2_total/ticker_total*100:.1f}%)")
                print(f"    - Stage 3 (Output): {stage3_total:.2f}s ({stage3_total/ticker_total*100:.1f}%)")
                print(f"    - Total time: {ticker_total:.2f} seconds")
                
                processed_count += 1
            else:
                # Validation failed - skip output generation
                print(f"\n❌ SKIPPING OUTPUT GENERATION")
                print(f"  Validation failed for {ticker.upper()}")
                print(f"  Only detection results saved: {os.path.join(results_path, f'{ticker}_detected_compliments_before_validation.json')}")
                
                # Ticker Summary for failed validation
                ticker_end_time = time.time()
                ticker_total = ticker_end_time - ticker_start_time
                
                print(f"\n❌ TICKER {ticker.upper()} VALIDATION FAILED:")
                print(f"  📊 Statistics:")
                print(f"    - Total quarters processed: {len(all_quarter_compliments)}")
                print(f"    - Potential compliments detected: {potential_compliments_count}")
                print(f"    - Validation status: FAILED - JSON parsing error")
                print(f"  ⏱️  Timing breakdown:")
                print(f"    - Stage 1 (Detection): {stage1_total:.2f}s ({stage1_total/ticker_total*100:.1f}%)")
                print(f"    - Stage 2 (Validation): {stage2_total:.2f}s ({stage2_total/ticker_total*100:.1f}%)")
                print(f"    - Total time: {ticker_total:.2f} seconds")
                
                failed_count += 1

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            failed_count += 1

    # Overall Summary
    overall_end_time = time.time()
    overall_total = overall_end_time - overall_start_time
    
    print(f"\n{'='*100}")
    print(f"🏁 OVERALL PROCESSING SUMMARY")
    print(f"{'='*100}")
    print(f"📈 Results:")
    print(f"  - Total tickers processed: {processed_count}/{len(target_tickers)}")
    print(f"  - Successfully completed: {processed_count}")
    print(f"  - Failed: {failed_count}")
    print(f"⏱️  Performance:")
    print(f"  - Total processing time: {overall_total:.2f} seconds ({overall_total/60:.1f} minutes)")
    print(f"  - Average time per ticker: {overall_total/len(target_tickers):.2f} seconds")
    print(f"  - Total API calls made: {total_api_calls}")
    print(f"📁 Output location: {results_path}")
    print(f"✅ Processing complete!")


def prepare_batch_validation_input(all_quarter_compliments):
    """Prepare batch validation input - send the same JSON structure as detection output."""
    # Remove original_text before sending to validation (not needed)
    validation_input = {}
    for quarter_key, quarter_data in all_quarter_compliments.items():
        validation_input[quarter_key] = {
            'year': quarter_data['year'],
            'quarter': quarter_data['quarter'],
            'date': quarter_data['date'],
            'compliments': quarter_data['compliments']
        }
    
    return validation_input


def parse_batch_validation_response(validation_response, original_structure):
    """Use the direct LLM validation response without any post-processing."""
    
    # Simply return the validation response as-is, trusting the LLM to do all the work
    if isinstance(validation_response, dict) and len(validation_response) > 0:
        print(f"    Using direct LLM validation output for {len(validation_response)} quarters...")
        return validation_response
    else:
        print("    Warning: No valid validation response received, keeping original structure")
        return original_structure


if __name__ == "__main__":
    main()