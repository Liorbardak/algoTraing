#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime
import sys
from activateChatty import ActivateChatty


def run_validation_for_stock_dual_field(stock_ticker, results_folder):
    """Run validation for a single stock with dual-field structure."""
    print(f"\n🔍 Processing validation for {stock_ticker}...")
    
    # Load the detected compliments (before validation)
    input_file = os.path.join(results_folder, f"{stock_ticker}_detected_compliments_before_validation.json")
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            detected_compliments = json.load(f)
        
        print(f"  📁 Loaded {len(detected_compliments)} quarters of data")
        

        
        # Process each quarter
        validated_compliments = {}
        chatty = ActivateChatty()
        
        for quarter_key, quarter_data in detected_compliments.items():
            print(f"  📅 Processing {quarter_key}...")
            

            
            # Get all analysts for this quarter
            analysts = quarter_data.get('compliments', [])
            print(f"    👥 Processing {len(analysts)} analysts for {quarter_key}...")
            
            # Call validation API with the entire quarter's data (same as extractComplimentsDualField)
            validation_start = time.time()
            validation_response = chatty.activate(
                '../../data/prompts/validation.txt',
                json.dumps(analysts),
                'gpt-4o'
            )
            validation_end = time.time()
            
            print(f"    ✅ Validation completed in {validation_end - validation_start:.2f} seconds")
            
            # Try to apply the validation response (same logic as extractComplimentsDualField)
            validated_analysts = None
            
            if isinstance(validation_response, list) and len(validation_response) == len(analysts):
                # If we got a list of the same length, use it directly
                validated_analysts = validation_response
            elif isinstance(validation_response, dict):
                # Check for different possible keys in the response
                if 'compliments' in validation_response:
                    validated_analysts = validation_response['compliments']
                elif 'analysts' in validation_response:
                    validated_analysts = validation_response['analysts']
                elif 'analyst_list' in validation_response:
                    validated_analysts = validation_response['analyst_list']
            
            if validated_analysts and len(validated_analysts) == len(analysts):
                # Update the analysts with validation results
                for i, validated_analyst in enumerate(validated_analysts):
                    if i < len(analysts):
                        # Update the level fields based on the validation response
                        for quote_num in [1, 2, 3]:
                            level_field = f'level_quote{quote_num}'
                            if level_field in validated_analyst:
                                old_value = analysts[i].get(level_field, 0)
                                new_value = validated_analyst[level_field]
                                analysts[i][level_field] = new_value
                                if old_value != new_value:
                                    print(f"      Updated analyst {i+1} {level_field}: {old_value} -> {new_value}")
            else:
                print(f"      Could not parse validation response format: {type(validation_response)}")
                if validated_analysts:
                    print(f"      Expected {len(analysts)} analysts, got {len(validated_analysts)}")
            
            # Store the validated data for this quarter
            validated_compliments[quarter_key] = {
                'year': quarter_data.get('year'),
                'quarter': quarter_data.get('quarter'),
                'date': quarter_data.get('date'),
                'compliments': analysts
            }
        
        # Save the validated data with the same structure as extractComplimentsDualField output
        output_file = os.path.join(results_folder, f"{stock_ticker}_all_revalidated_compliments.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validated_compliments, f, indent=2)
        
        print(f"  💾 Saved validation results to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {stock_ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run validation for all stocks with dual-field structure."""
    print("🔄 Re-running validation for all stocks (dual-field structure)...")
    
    # Find the most recent results directory
    results_root = os.path.join('..', '..','..', '..','data', 'results')
    try:
        all_results_dirs = [d for d in os.listdir(results_root) 
                           if os.path.isdir(os.path.join(results_root, d)) 
                           and d.startswith('results_dual_field')]
        
        if not all_results_dirs:
            print("❌ No dual-field results directories found.")
            return
        
        # Get the most recent directory
        latest_dir_name = max(all_results_dirs)
        results_base = os.path.join(results_root, latest_dir_name)
        
    except FileNotFoundError:
        print(f"❌ Results directory not found: {results_root}")
        return

    print(f"📁 Using results folder: {results_base}")

    # Dynamically find tickers from filenames
    stock_tickers = []
    for filename in os.listdir(results_base):
        if filename.endswith("_detected_compliments_before_validation.json"):
            ticker = filename.replace("_detected_compliments_before_validation.json", "")
            stock_tickers.append(ticker)

    if not stock_tickers:
        print("❌ No tickers found to process in the specified directory.")
        return

    print(f"🎯 Found {len(stock_tickers)} tickers to process: {', '.join(stock_tickers)}")
    
    # Filter to only process specific tickers
    stock_tickers = ["ADMA",'ADM','ANSS','AXON',"BSX"]
    print(f"🎯 Processing only AJG, ADM, CYBR...")
    
    # Process each stock
    successful = 0
    failed = 0
    
    for ticker in stock_tickers:
        if run_validation_for_stock_dual_field(ticker, results_base):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Validation Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Results saved in: {results_base}")
    print(f"  📁 Raw responses saved in subdirectories for each ticker")

if __name__ == "__main__":
    main() 