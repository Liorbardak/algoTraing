import os
import json
import time
import copy
import shutil
from datetime import datetime
from activateChatty import ActivateChatty
from extractComplimentsFromEarningCalls import EarningsComplimentAnalyzer


class DualFieldComplimentAnalyzer(EarningsComplimentAnalyzer):
    """
    Extends the base analyzer to handle three-quote compliment structure:
    - level_quote1: for the first quote field
    - level_quote2: for the second quote field  
    - level_quote3: for the third quote field
    Each quote is validated independently and levels can be changed from 0 to 1.
    """
    
    def validate_and_update_compliments(self, compliments, activateChatty, prompt_validation_path, text=None):
        """
        Validates a list of compliments by sending them to the LLM and applying the response.
        
        Args:
            compliments (list): The list of compliment dictionaries to validate.
            activateChatty (ActivateChatty): The API wrapper.
            prompt_validation_path (str): Path to the validation prompt.
            text (str, optional): The original earnings call text. Not used in this version.
            
        Returns:
            list: The list of compliments with updated validation status.
        """
        print(f"  Validating {len(compliments)} detected compliments...")
        
        # Send the entire list of compliments to validation
        input_data = json.dumps(compliments, indent=2)
        print(f"    Sending to validation: {input_data[:500]}...")
        
        validation_response = activateChatty.activate(
            prompt_validation_path,
            input_data,
            "gpt-4o"
        )
        
        print(f"    LLM validation response: {validation_response}")
        print(f"    Response type: {type(validation_response)}")
        print(f"    Response length: {len(str(validation_response)) if validation_response else 0}")
        if isinstance(validation_response, str):
            print(f"    Response is string, content: '{validation_response}'")
        elif validation_response is None:
            print(f"    Response is None")
        elif isinstance(validation_response, list):
            print(f"    Response is list with {len(validation_response)} items")
        elif isinstance(validation_response, dict):
            print(f"    Response is dict with keys: {list(validation_response.keys())}")
        
        # Try to apply the validation response
        validated_analysts = None
        
        if isinstance(validation_response, list) and len(validation_response) == len(compliments):
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
        
        if validated_analysts and len(validated_analysts) == len(compliments):
            # Update the compliments with validation results
            for i, validated_analyst in enumerate(validated_analysts):
                if i < len(compliments):
                    # Update the level fields based on the validation response
                    for quote_num in [1, 2, 3]:
                        level_field = f'level_quote{quote_num}'
                        if level_field in validated_analyst:
                            old_value = compliments[i].get(level_field, 0)
                            new_value = validated_analyst[level_field]
                            compliments[i][level_field] = new_value
                            if old_value != new_value:
                                print(f"      Updated analyst {i+1} {level_field}: {old_value} -> {new_value}")
        else:
            print(f"      Could not parse validation response format: {type(validation_response)}")
            if validated_analysts:
                print(f"      Expected {len(compliments)} analysts, got {len(validated_analysts)}")
        
        return compliments
    
    def check_compliment_in_text(self, compliment, text):
        """
        Override the parent method to check if any of the three quotes are present in the text.
        
        Args:
            compliment (dict): The compliment dictionary with quote1, quote2, quote3 fields
            text (str): The original earnings call text
            
        Returns:
            bool: True if any quote is found in the text, False otherwise
        """
        import re
        
        # Check each quote field
        for quote_num in [1, 2, 3]:
            quote_field = f'quote{quote_num}'
            quote_text = compliment.get(quote_field, "")
            
            if not quote_text.strip():
                continue
                
            # Normalize the quote text for comparison
            normalized_quote = quote_text.lower()
            normalized_quote = re.sub(r'[.,] ', '', normalized_quote).strip()
            normalized_quote = normalized_quote.replace(" ", "")
            normalized_quote = normalized_quote.replace(".", "")
            normalized_quote = normalized_quote.replace(",", "")

            # Check if quote is in the text
            earning_text = self.normalize_text(text)

            # Look for a substantial portion of the quote in the text
            if len(normalized_quote) > 1 and normalized_quote[1:min(len(normalized_quote), 30)] in earning_text:
                # Filter out simple "thanks" or "great" which aren't substantial compliments
                words = normalized_quote.split()
                if not (len(words) < 3 and (any(word.lower() == "thanks" for word in words) or
                                            any(word.lower() == "great" for word in words))):
                    return True

        return False


def main():
    """
    Main function to process earnings calls with dual-field compliment structure.
    """
    # --- Configuration ---
    data_path = '../../../../data/'
    tickers_path = '../../../../data/tickers/'
    prompt_path = '../../data/prompts/'
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_folder_name = f"results_dual_field_{timestamp}"
    results_path = f'../../../../data/results/{results_folder_name}/'
    
    prompt_detection_path = os.path.join(prompt_path, 'detection.txt')
    prompt_validation_path = os.path.join(prompt_path, 'validation.txt')
    
    analyzer = DualFieldComplimentAnalyzer(data_path, tickers_path, results_path)
    chatty = ActivateChatty()
    
    # Run only for ADMA as requested
    target_tickers = ["ADMA","ADM","AJG","ANSS","AXON","CLBT","CYBR","BSX"]
    #target_tickers = ["AMRK", "ANET", "CAMT", "FIX", "SMCI", "FULC", "DECK", "BR"]
    print(f"Processing {len(target_tickers)} tickers with dual-field validation...")
    
    # --- Setup Results Directory ---
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    try:
        shutil.copy2(prompt_detection_path, os.path.join(results_path, 'detection_prompt_used.txt'))
        shutil.copy2(prompt_validation_path, os.path.join(results_path, 'validation_prompt_used.txt'))
        print(f"✓ Copied prompts to {results_path}")
    except Exception as e:
        print(f"Warning: Could not copy prompt files: {e}")

    overall_start_time = time.time()
    
    # --- Main Processing Loop ---
    for ticker in target_tickers:
        print(f"\n{'='*80}\nProcessing Ticker: {ticker.upper()}\n{'='*80}")
        
        earnings = analyzer.load_earnings_data(os.path.join(tickers_path, ticker), ticker)
        valid_earnings = [call for call in earnings if call.get('text') and call.get('parsed_text')]

        if not valid_earnings:
            print(f"No valid earnings data found for {ticker}.")
            continue

        all_detected_compliments = {}
        all_validated_compliments = {}

        for call in valid_earnings:
            quarter_key = f"{call['year']}_Q{call['quarter']}"
            
            print(f"\nProcessing {ticker} for {quarter_key}...")

            # --- Stage 1: Detection ---
            print("  Detecting compliments...")
            tentative_compliments = chatty.activate(
                prompt_detection_path,
                call['parsed_text'],
                "gpt-4.1-mini"
            )
            
            if not isinstance(tentative_compliments, list):
                tentative_compliments = []
            
            # Store results before validation
            date = call.get('date', f"{call['year']}-{call['quarter']*3:02d}-01")
            all_detected_compliments[quarter_key] = {
                'year': call['year'], 'quarter': call['quarter'], 'date': date,
                'compliments': tentative_compliments
            }

            # --- Stage 2: Validation ---
            compliments_to_validate = copy.deepcopy(tentative_compliments)
            
            validated_compliments = analyzer.validate_and_update_compliments(
                compliments_to_validate, chatty, prompt_validation_path, call['text']
            )
            
            # Store results after validation
            all_validated_compliments[quarter_key] = {
                'year': call['year'], 'quarter': call['quarter'], 'date': date,
                'compliments': validated_compliments
            }

        # --- Stage 3: Save Output Files ---
        print(f"\nSaving output files for {ticker}...")

        # File 1: Before Validation
        before_val_file = os.path.join(results_path, f'{ticker}_detected_compliments_before_validation.json')
        with open(before_val_file, 'w', encoding='utf-8') as f:
            json.dump(all_detected_compliments, f, ensure_ascii=False, indent=4)
        print(f"  ✓ Saved before-validation data to {before_val_file}")

        # File 2: After Validation
        after_val_file = os.path.join(results_path, f'{ticker}_all_validated_compliments.json')
        with open(after_val_file, 'w', encoding='utf-8') as f:
            json.dump(all_validated_compliments, f, ensure_ascii=False, indent=4)
        print(f"  ✓ Saved after-validation data to {after_val_file}")

        # File 3: Summary Statistics (updated for three-quote structure)
        aggregate_results = []
        for q_key, q_data in all_validated_compliments.items():
            compliments = q_data['compliments']
            
            # Count analysts with compliments (any quote level is 1)
            analysts_with_compliments = sum(1 for c in compliments 
                                          if c.get('level_quote1', 0) == 1 or 
                                             c.get('level_quote2', 0) == 1 or 
                                             c.get('level_quote3', 0) == 1)
            
            # Count analysts with quote1 compliments
            analysts_with_quote1_compliments = sum(1 for c in compliments 
                                                 if c.get('level_quote1', 0) == 1)
            
            # Count analysts with quote2 compliments
            analysts_with_quote2_compliments = sum(1 for c in compliments 
                                                 if c.get('level_quote2', 0) == 1)
            
            # Count analysts with quote3 compliments
            analysts_with_quote3_compliments = sum(1 for c in compliments 
                                                 if c.get('level_quote3', 0) == 1)
            
            summary = {
                'date': q_data['date'],
                'quarter': q_key,
                'total_number_of_analysts': len(compliments),
                'number_of_analysts_with_compliments': analysts_with_compliments,
                'number_of_analysts_with_quote1_compliments': analysts_with_quote1_compliments,
                'number_of_analysts_with_quote2_compliments': analysts_with_quote2_compliments,
                'number_of_analysts_with_quote3_compliments': analysts_with_quote3_compliments
            }
            aggregate_results.append(summary)
        
        summary_file = os.path.join(results_path, f'{ticker}_compliment_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(aggregate_results, f, ensure_ascii=False, indent=4)
        print(f"  ✓ Saved summary statistics to {summary_file}")

    overall_end_time = time.time()
    print(f"\n🏁 Total processing time: {overall_end_time - overall_start_time:.2f} seconds")


if __name__ == "__main__":
    main() 