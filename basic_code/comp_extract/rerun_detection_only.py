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
from extractComplimentsFromEarningCalls import EarningsComplimentAnalyzer
from datetime import datetime
import shutil


def main():
    """
    Main function to process earnings calls and extract analyst compliments.
    Modified with batch processing, timing measurements, and optimized output.
    """
    import time

    # Configuration paths
    basedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    data_path = os.path.join(basedir, 'data')
    tickers_path = os.path.join(data_path, 'tickers')

    # Create timestamp-based results folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder_name = f"results_{timestamp}"
    # results_path = f'../../data/results/{results_folder_name}/'
    results_path = os.path.join(data_path,'results', results_folder_name)
    prompt_path = '../../data/prompts/'

    # Prompt file paths
    #prompt_detection_path = prompt_path + 'detection.txt'
    prompt_detection_path = 'prompts/detection_prompt_all_sentences.txt'
    #prompt_detection_path = 'prompts/detection_prompt_first_two_sentences_and_last_one.txt'
    prompt_validation_path = prompt_path + 'validation.txt'

    # Initialize the analyzer classes
    analyzer = EarningsComplimentAnalyzer(data_path, tickers_path, results_path)
    chatty = ActivateChatty()

    # Process these tickers
    # target_tickers = ["ACAD", "ACB", "ACHC", "ADMA", "AES", "AKAM", "AMRC", "AMRN", "AMSC", "AMPX", "ANF", "AQST", "ASLE", "ASTH", "AVAH", "AVDX", "AVGO", "AVPT", "BKE", "BKTI", "BBCP", "BMA", "BNS", "BWAY", "BBAR", "BTBT", "CAPL", "CAR", "CCBS", "CCBG", "CERS", "CIFR", "CLS", "COIN", "CRDO", "CRS", "CSAN", "CSIQ", "DAKT", "DAY", "DIN", "DRD", "DSP", "DUOL", "EAT", "EIC", "EDN", "ESOA", "ETN", "ETSY", "EXFY", "EXK", "FBMS", "FGBI", "FICO", "FIVE", "FMC", "FMAO", "FNB", "FORR", "FRPH", "FSM", "FTK", "FTAI", "GHM", "GMAB", "GMED", "GNP", "GOSS", "GPN", "HEPS", "HBT", "HL", "HLF", "HNVR", "HOOD", "HRTG", "IART", "IESC", "IREN", "INMD", "IRWD", "KEQU", "KINS", "KOPN", "LFVN", "LMB", "LOCO", "LULU", "LWAY", "MAMA", "MARA", "MASI", "META", "MET", "MHK", "MLCO", "MMYT", "MOD", "MODG", "MOV", "MRAM", "MPTI", "MTRN", "MYO", "MYPS", "MNTK", "NBTA", "NBTB", "NFE", "NGVC", "NL", "NOVT", "NNI", "NRC", "NXT", "ODP", "OFLX", "OKTA", "OLP", "OPK", "OSCR", "OTRK", "OZK", "PAMT", "PBA", "PAY", "PCB", "PEBO", "PERF", "PLTR", "PM", "POOL", "PSFE", "PSIX", "PSX", "PUBM", "POWL", "PRU", "RCL", "RDWR", "REAX", "RGLD", "RKLB", "RRBI", "RRR", "SFL", "SGML", "SHEL", "SHYF", "SITC", "SITE", "SKYW", "SLQT", "SMBC", "SMBK", "SOL", "SPH", "SPOT", "SPTN", "SSTI", "STAA", "STBA", "STRL", "SU", "SUN", "SVV", "SWBI", "SYBT", "TALK", "TATT", "TBLA", "TGTX", "TOWN", "TRP", "TSEM", "TS", "TSCO", "TV", "TZOO", "UL", "UGP", "UTI", "UVSP", "VOYA", "VRT", "VST", "WB", "WBS", "WD", "WDH", "WOOF", "WULF", "XPO", "YPF", "ZI", "ZION"]

    # target_tickers = ["CYBR"]
    #target_tickers = ["ADM"]
    target_tickers = ['ADMA','CLBT']
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
        print(f"\n{'=' * 80}")
        print(f"Processing ticker {i}/{len(target_tickers)}: {ticker.upper()}")
        print(f"{'=' * 80}")
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
                    date = f"{call['year']}-{call['quarter'] * 3:02d}-01"

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

                print(
                    f"    Found {len([c for c in tentative_compliments if int(c.get('level', 0)) > 0])} potential compliments")

            stage1_end = time.time()
            stage1_total = stage1_end - stage1_start

            print(f"\nSTAGE 1 COMPLETED:")
            print(f"  Total time: {stage1_total:.2f} seconds")
            print(f"  API calls time: {stage1_api_time:.2f} seconds ({stage1_api_time / stage1_total * 100:.1f}%)")
            print(
                f"  Processing time: {stage1_processing_time:.2f} seconds ({stage1_processing_time / stage1_total * 100:.1f}%)")
            print(f"  Other operations: {stage1_total - stage1_api_time - stage1_processing_time:.2f} seconds")

            # Save detected compliments before validation
            print(f"\nSaving detected compliments before validation...")
            detection_only_filename = os.path.join(results_path,
                                                   f'{ticker}_detected_compliments_before_validation.json')

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

        except Exception as e:
            print(f"❌ Error processing {ticker}: {e}")
            failed_count += 1


    # Overall Summary
    overall_end_time = time.time()
    overall_total = overall_end_time - overall_start_time

    print(f"\n{'=' * 100}")
    print(f"🏁 OVERALL PROCESSING SUMMARY")
    print(f"{'=' * 100}")
    print(f"📈 Results:")
    print(f"  - Total tickers processed: {processed_count}/{len(target_tickers)}")
    print(f"  - Successfully completed: {processed_count}")
    print(f"  - Failed: {failed_count}")
    print(f"⏱️  Performance:")
    print(f"  - Total processing time: {overall_total:.2f} seconds ({overall_total / 60:.1f} minutes)")
    print(f"  - Average time per ticker: {overall_total / len(target_tickers):.2f} seconds")
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

