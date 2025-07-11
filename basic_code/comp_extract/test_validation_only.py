#!/usr/bin/env python3
"""
Test script to isolate and debug the validation stage only.
This script loads the detected compliments from the JSON file and tests the validation process.
"""

import json
import os
import sys
from activateChatty import ActivateChatty

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

def main():
    print("=" * 80)
    print("VALIDATION STAGE TEST - ISOLATED")
    print("=" * 80)
    
    # Configuration
    prompt_path = '../../data/prompts/'
    prompt_validation_path = prompt_path + 'validation.txt'
    
    # Find the most recent results folder
    results_base_path = '../../data/results/'
    results_folders = [f for f in os.listdir(results_base_path) if f.startswith('results_')]
    results_folders.sort(reverse=True)  # Get the most recent one
    
    if not results_folders:
        print("❌ No results folders found!")
        return
    
    latest_results_folder = results_folders[0]
    results_path = os.path.join(results_base_path, latest_results_folder)
    
    print(f"Using results folder: {latest_results_folder}")
    
    # Load the detected compliments (before validation)
    detected_file = os.path.join(results_path, 'ADMA_detected_compliments_before_validation.json')
    
    if not os.path.exists(detected_file):
        print(f"❌ File not found: {detected_file}")
        return
    
    print(f"Loading detected compliments from: {detected_file}")
    
    with open(detected_file, 'r', encoding='utf-8') as f:
        all_quarter_compliments = json.load(f)
    
    print(f"✓ Loaded {len(all_quarter_compliments)} quarters of data")
    
    # Count potential compliments
    potential_compliments_count = 0
    for quarter_key, quarter_data in all_quarter_compliments.items():
        compliments = quarter_data.get('compliments', [])
        quarter_potential = sum(1 for c in compliments if int(c.get('level', 0)) > 0)
        potential_compliments_count += quarter_potential
        print(f"  {quarter_key}: {len(compliments)} analysts, {quarter_potential} potential compliments")
    
    print(f"Total potential compliments to validate: {potential_compliments_count}")
    
    if potential_compliments_count == 0:
        print("❌ No potential compliments found to validate!")
        return
    
    print("\n" + "=" * 50)
    print("STARTING VALIDATION TEST")
    print("=" * 50)
    
    # Initialize ChatGPT API
    chatty = ActivateChatty()
    
    # Prepare batch validation input
    print("Step 1: Preparing batch validation input...")
    batch_validation_input = prepare_batch_validation_input(all_quarter_compliments)
    
    print(f"  Input type: {type(batch_validation_input)}")
    print(f"  Input quarters: {len(batch_validation_input)}")
    
    # Convert to JSON string
    batch_validation_input_str = json.dumps(batch_validation_input, ensure_ascii=False, indent=2)
    print(f"  JSON string length: {len(batch_validation_input_str)} characters")
    
    # Show a sample of the input
    print("\nSample of validation input (first 500 chars):")
    print(batch_validation_input_str[:500] + "...")
    
    print("\nStep 2: Sending to validation API...")
    
    # Make the API call
    try:
        batch_validated_response = chatty.activate(
            prompt_validation_path,
            batch_validation_input_str,
            "gpt-4o"
        )
        
        print(f"✓ API call completed successfully!")
        print(f"  Response type: {type(batch_validated_response)}")
        
        if isinstance(batch_validated_response, dict):
            print(f"  Response has {len(batch_validated_response)} top-level keys")
            print(f"  Response keys: {list(batch_validated_response.keys())}")
            
            # Debug: Show the actual response structure
            print(f"\nFull response structure:")
            for key, value in batch_validated_response.items():
                print(f"  {key}: {type(value)}")
                if isinstance(value, list):
                    print(f"    List length: {len(value)}")
                    if len(value) > 0:
                        print(f"    First item type: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"    First item keys: {list(value[0].keys())}")
                elif isinstance(value, dict):
                    print(f"    Dict keys: {list(value.keys())}")
            
            # Show raw response (first 1000 chars)
            response_str = json.dumps(batch_validated_response, indent=2)
            print(f"\nRaw response (first 1000 chars):")
            print(response_str[:1000] + "..." if len(response_str) > 1000 else response_str)
        
        elif isinstance(batch_validated_response, list):
            print(f"  Response is a list with {len(batch_validated_response)} items")
        else:
            print(f"  Unexpected response type: {type(batch_validated_response)}")
            print(f"  Response: {batch_validated_response}")
    
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return
    
    print("\nStep 3: Parsing validation response...")
    
    # Parse the response
    try:
        validated_quarter_compliments = parse_batch_validation_response(
            batch_validated_response, all_quarter_compliments
        )
        
        print("✓ Response parsed successfully!")
        
        # Count remaining compliments after validation
        remaining_compliments = 0
        for quarter_key, quarter_data in validated_quarter_compliments.items():
            compliments = quarter_data.get('compliments', [])
            quarter_remaining = sum(1 for c in compliments if int(c.get('level', 0)) > 0)
            remaining_compliments += quarter_remaining
            print(f"  {quarter_key}: {quarter_remaining} compliments remaining after validation")
        
        print(f"Total remaining compliments: {remaining_compliments}")
        
        # Save the validated results
        validated_file = os.path.join(results_path, 'ADMA_test_validated_compliments.json')
        with open(validated_file, 'w', encoding='utf-8') as f:
            json.dump(validated_quarter_compliments, f, ensure_ascii=False, indent=4)
        
        print(f"✓ Validated results saved to: {validated_file}")
        
        # Show validation success/failure details
        print("\n" + "=" * 50)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 50)
        print(f"Original potential compliments: {potential_compliments_count}")
        print(f"Remaining after validation: {remaining_compliments}")
        print(f"Rejected by validation: {potential_compliments_count - remaining_compliments}")
        
        if remaining_compliments > 0:
            print("\n✅ VALIDATION WORKING - Some compliments passed validation!")
        else:
            print("\n⚠️  ALL COMPLIMENTS REJECTED - Check validation prompt/logic")
    
    except Exception as e:
        print(f"❌ Response parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 