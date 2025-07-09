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
    #prompt_detection_path = 'prompts/detection_prompt_all_sentences.txt'
    prompt_detection_path = 'prompts/detection_prompt_first_two_sentences_and_last_one.txt'
    prompt_validation_path = prompt_path + 'validation.txt'

    # Initialize the analyzer classes
    analyzer = EarningsComplimentAnalyzer(data_path, tickers_path, results_path)
    chatty = ActivateChatty()

    # Process these tickers
    # target_tickers = ["ACAD", "ACB", "ACHC", "ADMA", "AES", "AKAM", "AMRC", "AMRN", "AMSC", "AMPX", "ANF", "AQST", "ASLE", "ASTH", "AVAH", "AVDX", "AVGO", "AVPT", "BKE", "BKTI", "BBCP", "BMA", "BNS", "BWAY", "BBAR", "BTBT", "CAPL", "CAR", "CCBS", "CCBG", "CERS", "CIFR", "CLS", "COIN", "CRDO", "CRS", "CSAN", "CSIQ", "DAKT", "DAY", "DIN", "DRD", "DSP", "DUOL", "EAT", "EIC", "EDN", "ESOA", "ETN", "ETSY", "EXFY", "EXK", "FBMS", "FGBI", "FICO", "FIVE", "FMC", "FMAO", "FNB", "FORR", "FRPH", "FSM", "FTK", "FTAI", "GHM", "GMAB", "GMED", "GNP", "GOSS", "GPN", "HEPS", "HBT", "HL", "HLF", "HNVR", "HOOD", "HRTG", "IART", "IESC", "IREN", "INMD", "IRWD", "KEQU", "KINS", "KOPN", "LFVN", "LMB", "LOCO", "LULU", "LWAY", "MAMA", "MARA", "MASI", "META", "MET", "MHK", "MLCO", "MMYT", "MOD", "MODG", "MOV", "MRAM", "MPTI", "MTRN", "MYO", "MYPS", "MNTK", "NBTA", "NBTB", "NFE", "NGVC", "NL", "NOVT", "NNI", "NRC", "NXT", "ODP", "OFLX", "OKTA", "OLP", "OPK", "OSCR", "OTRK", "OZK", "PAMT", "PBA", "PAY", "PCB", "PEBO", "PERF", "PLTR", "PM", "POOL", "PSFE", "PSIX", "PSX", "PUBM", "POWL", "PRU", "RCL", "RDWR", "REAX", "RGLD", "RKLB", "RRBI", "RRR", "SFL", "SGML", "SHEL", "SHYF", "SITC", "SITE", "SKYW", "SLQT", "SMBC", "SMBK", "SOL", "SPH", "SPOT", "SPTN", "SSTI", "STAA", "STBA", "STRL", "SU", "SUN", "SVV", "SWBI", "SYBT", "TALK", "TATT", "TBLA", "TGTX", "TOWN", "TRP", "TSEM", "TS", "TSCO", "TV", "TZOO", "UL", "UGP", "UTI", "UVSP", "VOYA", "VRT", "VST", "WB", "WBS", "WD", "WDH", "WOOF", "WULF", "XPO", "YPF", "ZI", "ZION"]

    # target_tickers = ["CYBR"]
    #target_tickers = ["ADM"]
    target_tickers = ['ADMA']
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

        #     # STAGE 2: Batch Validation (gpt-4o)
        #     print(f"\nSTAGE 2: Batch Validation Phase")
        #     print("-" * 50)
        #     stage2_start = time.time()
        #
        #     # Count potential compliments before batch validation
        #     potential_compliments_count = sum(
        #         len([c for c in qd['compliments'] if int(c.get('level', 0)) > 0])
        #         for qd in all_quarter_compliments.values()
        #     )
        #     print(f"  Found {potential_compliments_count} potential compliments to validate")
        #
        #     stage2_api_time = 0
        #     stage2_processing_time = 0
        #     validation_successful = False
        #
        #     if potential_compliments_count > 0:
        #         # Prepare batch validation input
        #         prep_start = time.time()
        #         batch_validation_input = prepare_batch_validation_input(all_quarter_compliments)
        #         prep_end = time.time()
        #
        #         print(f"  Batch preparation completed in {prep_end - prep_start:.2f} seconds")
        #
        #         # Convert dictionary to JSON string for the API call
        #         try:
        #             batch_validation_input_str = json.dumps(batch_validation_input, ensure_ascii=False, indent=2)
        #
        #             # Single API call for all verification
        #             api_start = time.time()
        #             batch_validated_response = chatty.activate(
        #                 prompt_validation_path,
        #                 batch_validation_input_str,
        #                 "gpt-4o"
        #             )
        #             api_end = time.time()
        #             stage2_api_time = api_end - api_start
        #             total_api_calls += 1
        #             print(f"  Batch validation API call completed in {stage2_api_time:.2f} seconds")
        #
        #             # Parse batch validation response
        #             parse_start = time.time()
        #             if isinstance(batch_validated_response, dict) and len(batch_validated_response) > 0:
        #                 validated_quarter_compliments = parse_batch_validation_response(
        #                     batch_validated_response, all_quarter_compliments
        #                 )
        #                 validation_successful = True
        #                 print(f"  ✅ Validation successful with {len(batch_validated_response)} quarters processed")
        #             elif isinstance(batch_validated_response, list) and len(batch_validated_response) > 0:
        #                 validated_quarter_compliments = parse_batch_validation_response(
        #                     batch_validated_response, all_quarter_compliments
        #                 )
        #                 validation_successful = True
        #                 print(f"  ✅ Validation successful with {len(batch_validated_response)} validation results")
        #             else:
        #                 validated_quarter_compliments = all_quarter_compliments
        #                 validation_successful = False
        #                 print(f"  ❌ Validation failed - JSON parsing error or empty response")
        #             parse_end = time.time()
        #             stage2_processing_time = (prep_end - prep_start) + (parse_end - parse_start)
        #
        #             # Verify validation was applied
        #             validation_applied_count = sum(
        #                 len([c for c in qd['compliments'] if int(c.get('level', 0)) > 0])
        #                 for qd in validated_quarter_compliments.values()
        #             )
        #             print(f"  Validation completed. Remaining valid compliments: {validation_applied_count}")
        #
        #         except Exception as e:
        #             print(f"  ❌ Error in validation process: {e}")
        #             validation_successful = False
        #             validated_quarter_compliments = all_quarter_compliments
        #             stage2_api_time = 0
        #             stage2_processing_time = (prep_end - prep_start)
        #     else:
        #         print("  No compliments found to validate")
        #         validated_quarter_compliments = all_quarter_compliments
        #         validation_successful = True  # No validation needed, so consider it successful
        #
        #     stage2_end = time.time()
        #     stage2_total = stage2_end - stage2_start
        #
        #     print(f"\nSTAGE 2 COMPLETED:")
        #     print(f"  Total time: {stage2_total:.2f} seconds")
        #     print(f"  API calls time: {stage2_api_time:.2f} seconds ({stage2_api_time / stage2_total * 100:.1f}%)")
        #     print(
        #         f"  Processing time: {stage2_processing_time:.2f} seconds ({stage2_processing_time / stage2_total * 100:.1f}%)")
        #
        #     # STAGE 2.5: Remove duplicate analysts (keeping only the highest level compliment per analyst)
        #     print(f"\nRemoving duplicate analysts...")
        #     dedup_start = time.time()
        #
        #     for quarter_key, quarter_data in validated_quarter_compliments.items():
        #         original_count = len(quarter_data['compliments'])
        #         quarter_data['compliments'] = analyzer.remove_duplications(quarter_data['compliments'])
        #         dedup_count = len(quarter_data['compliments'])
        #         if original_count != dedup_count:
        #             print(f"  {quarter_key}: Removed {original_count - dedup_count} duplicate analysts")
        #
        #     dedup_end = time.time()
        #     print(f"✓ Duplicate removal completed in {dedup_end - dedup_start:.2f} seconds")
        #
        #     # STAGE 3: Generate Output Files (only if validation was successful)
        #     if validation_successful:
        #         print(f"\nSTAGE 3: Output Generation")
        #         print("-" * 50)
        #         stage3_start = time.time()
        #
        #         # Remove original_text before saving (not needed in output)
        #         for quarter_data in validated_quarter_compliments.values():
        #             quarter_data.pop('original_text', None)
        #
        #         # File 1: All compliments from all quarters
        #         consolidated_filename = os.path.join(results_path, f'{ticker}_all_validated_compliments.json')
        #         with open(consolidated_filename, 'w', encoding='utf-8') as f:
        #             json.dump(validated_quarter_compliments, f, ensure_ascii=False, indent=4)
        #         print(f"  ✓ Saved consolidated compliments: {consolidated_filename}")
        #
        #         # File 2: Summary statistics
        #         aggregate_results = []
        #         total_analysts = 0
        #         total_valid_compliments = 0
        #
        #         for quarter_key, quarter_data in validated_quarter_compliments.items():
        #             compliments = quarter_data['compliments']
        #             date = quarter_data['date']
        #
        #             # Count analysts and compliments
        #             quarter_analysts = len(compliments)
        #             quarter_level_1 = sum(1 for c in compliments if int(c.get('level', 0)) == 1)
        #             quarter_level_2 = sum(1 for c in compliments if int(c.get('level', 0)) == 2)
        #             quarter_level_3 = sum(1 for c in compliments if int(c.get('level', 0)) == 3)
        #
        #             total_analysts += quarter_analysts
        #             total_valid_compliments += (quarter_level_1 + quarter_level_2 + quarter_level_3)
        #
        #             summary_stats = {
        #                 'date': date,
        #                 'quarter': quarter_key,
        #                 'total_number_of_analysts': quarter_analysts,
        #                 'number_of_analysts_comp_1': quarter_level_1,
        #                 'number_of_analysts_comp_2': quarter_level_2,
        #                 'number_of_analysts_comp_3': quarter_level_3
        #             }
        #             aggregate_results.append(summary_stats)
        #
        #         summary_filename = os.path.join(results_path, f'{ticker}_compliment_summary.json')
        #         with open(summary_filename, 'w', encoding='utf-8') as f:
        #             json.dump(aggregate_results, f, ensure_ascii=False, indent=4)
        #         print(f"  ✓ Saved summary statistics: {summary_filename}")
        #
        #         stage3_end = time.time()
        #         stage3_total = stage3_end - stage3_start
        #
        #         print(f"\nSTAGE 3 COMPLETED:")
        #         print(f"  Total time: {stage3_total:.2f} seconds")
        #
        #         # Ticker Summary
        #         ticker_end_time = time.time()
        #         ticker_total = ticker_end_time - ticker_start_time
        #
        #         print(f"\n🎉 TICKER {ticker.upper()} COMPLETED:")
        #         print(f"  📊 Statistics:")
        #         print(f"    - Total quarters processed: {len(validated_quarter_compliments)}")
        #         print(f"    - Total analysts: {total_analysts}")
        #         print(f"    - Total valid compliments: {total_valid_compliments}")
        #         print(f"  ⏱️  Timing breakdown:")
        #         print(f"    - Stage 1 (Detection): {stage1_total:.2f}s ({stage1_total / ticker_total * 100:.1f}%)")
        #         print(f"    - Stage 2 (Validation): {stage2_total:.2f}s ({stage2_total / ticker_total * 100:.1f}%)")
        #         print(f"    - Stage 3 (Output): {stage3_total:.2f}s ({stage3_total / ticker_total * 100:.1f}%)")
        #         print(f"    - Total time: {ticker_total:.2f} seconds")
        #
        #         processed_count += 1
        #     else:
        #         # Validation failed - skip output generation
        #         print(f"\n❌ SKIPPING OUTPUT GENERATION")
        #         print(f"  Validation failed for {ticker.upper()}")
        #         print(
        #             f"  Only detection results saved: {os.path.join(results_path, f'{ticker}_detected_compliments_before_validation.json')}")
        #
        #         # Ticker Summary for failed validation
        #         ticker_end_time = time.time()
        #         ticker_total = ticker_end_time - ticker_start_time
        #
        #         print(f"\n❌ TICKER {ticker.upper()} VALIDATION FAILED:")
        #         print(f"  📊 Statistics:")
        #         print(f"    - Total quarters processed: {len(all_quarter_compliments)}")
        #         print(f"    - Potential compliments detected: {potential_compliments_count}")
        #         print(f"    - Validation status: FAILED - JSON parsing error")
        #         print(f"  ⏱️  Timing breakdown:")
        #         print(f"    - Stage 1 (Detection): {stage1_total:.2f}s ({stage1_total / ticker_total * 100:.1f}%)")
        #         print(f"    - Stage 2 (Validation): {stage2_total:.2f}s ({stage2_total / ticker_total * 100:.1f}%)")
        #         print(f"    - Total time: {ticker_total:.2f} seconds")
        #
        #         failed_count += 1
        #
        # except Exception as e:
        #     print(f"❌ Error processing {ticker}: {e}")
        #     failed_count += 1

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

