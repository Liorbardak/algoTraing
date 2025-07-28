#!/usr/bin/env python3

import os
import json
import time
from datetime import datetime
import sys
from activateChatty import ActivateChatty


def run_validation_for_stock(stock_ticker, results_folder):
    """Run validation for a single stock."""
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
        
        # Convert to JSON string for API
        json_input = json.dumps(detected_compliments, indent=2)
        
        print(f"  🤖 Sending to validation API...")
        validation_start = time.time()
        
        # Initialize the API wrapper and call validation
        chatty = ActivateChatty()
        validation_response = chatty.activate(
            '../../data/prompts/validation.txt', 
            json_input, 
            'gpt-4o'
        )
        
        validation_end = time.time()
        print(f"  ✅ Validation completed in {validation_end - validation_start:.2f} seconds")
        
        # Parse the response
        try:
            if isinstance(validation_response, str):
                validated_data = json.loads(validation_response)
            else:
                validated_data = validation_response
            
            # Save the validation results
            output_file = os.path.join(results_folder, f"{stock_ticker}_revalidated_compliments.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validated_data, f, indent=2)
            
            print(f"  💾 Saved validation results to: {output_file}")
            
            # Save raw validation response for debugging
            raw_output_file = os.path.join(results_folder, f"{stock_ticker}_raw_validation_response.json")
            with open(raw_output_file, 'w', encoding='utf-8') as f:
                if isinstance(validation_response, str):
                    f.write(validation_response)
                else:
                    json.dump(validation_response, f, indent=2)
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse validation response: {e}")
            # Save the raw response for debugging
            error_file = os.path.join(results_folder, f"{stock_ticker}_validation_error.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(str(validation_response))
            return False
            
    except Exception as e:
        print(f"❌ Error processing {stock_ticker}: {e}")
        return False

def main():
    """Main function to run validation for all stocks."""
    print("🔄 Re-running validation for all stocks...")
    
    results_base = os.path.join('..', '..','..', '..','data', 'results','results_20250715_012407')
    
    if not os.path.exists(results_base):
        print(f"❌ Results directory not found: {results_base}")
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
    
    # Process each stock
    successful = 0
    failed = 0
    
    for ticker in stock_tickers:
        if run_validation_for_stock(ticker, results_base):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 Validation Summary:")
    print(f"  ✅ Successful: {successful}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Results saved in: {results_base}")

if __name__ == "__main__":
    main() 