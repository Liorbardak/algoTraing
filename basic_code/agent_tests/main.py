import os
import json
import pandas as pd
from detection_only import run_detection_only
from compare_compliments import run_evaluation

def analyze_errors(evaluation_path, tickers):
    """
    Analyze errors from the evaluation to understand what went wrong
    """
    error_analysis = {}
    
    for ticker in tickers:
        error_file = os.path.join(evaluation_path, f"{ticker}_compareToGT_level_errors.json")
        
        if not os.path.exists(error_file):
            print(f"Warning: Error file not found for {ticker}")
            continue
            
        with open(error_file, 'r', encoding='utf-8') as f:
            errors = json.load(f)
        
        ticker_analysis = {
            'total_errors': len(errors),
            'false_positives': 0,
            'false_negatives': 0,
            'error_patterns': [],
            'common_phrases': []
        }
        
        for error in errors:
            # Count false positives and false negatives
            if error['level'] == 1 and error['level_GT'] == 0:
                ticker_analysis['false_positives'] += 1
            elif error['level'] == 0 and error['level_GT'] == 1:
                ticker_analysis['false_negatives'] += 1
            
            # Analyze the quoted compliments to understand patterns
            if error['quoted_compliment']:
                ticker_analysis['error_patterns'].append({
                    'type': 'false_positive' if error['level'] == 1 and error['level_GT'] == 0 else 'false_negative',
                    'quoted': error['quoted_compliment'],
                    'gt_quoted': error['quoted_compliment_GT']
                })
        
        error_analysis[ticker] = ticker_analysis
    
    return error_analysis

def print_error_analysis(error_analysis):
    """
    Print a detailed analysis of the errors
    """
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    
    for ticker, analysis in error_analysis.items():
        print(f"\n{ticker} Analysis:")
        print(f"  Total Errors: {analysis['total_errors']}")
        print(f"  False Positives: {analysis['false_positives']}")
        print(f"  False Negatives: {analysis['false_negatives']}")
        
        if analysis['error_patterns']:
            print(f"  Error Patterns:")
            for pattern in analysis['error_patterns'][:5]:  # Show first 5 patterns
                print(f"    {pattern['type'].upper()}:")
                print(f"      Detected: '{pattern['quoted'][:100]}...'")
                if pattern['gt_quoted']:
                    print(f"      Ground Truth: '{pattern['gt_quoted'][:100]}...'")
                print()

def create_improved_prompt(original_prompt_path, error_analysis):
    """
    Create an improved prompt based on error analysis
    """
    with open(original_prompt_path, 'r', encoding='utf-8') as f:
        original_prompt = f.read()
    
    # Analyze common error patterns
    false_positive_phrases = []
    false_negative_phrases = []
    
    for ticker, analysis in error_analysis.items():
        for pattern in analysis['error_patterns']:
            if pattern['type'] == 'false_positive':
                false_positive_phrases.append(pattern['quoted'])
            elif pattern['type'] == 'false_negative':
                false_negative_phrases.append(pattern['quoted'])
    
    # Create improved prompt
    improved_prompt = original_prompt + "\n\n" + "IMPORTANT GUIDELINES BASED ON PREVIOUS ERRORS:\n\n"
    
    if false_positive_phrases:
        improved_prompt += "AVOID FALSE POSITIVES - Do NOT classify these as compliments:\n"
        improved_prompt += "- General greetings or polite responses (e.g., 'Thank you', 'Great, thank you')\n"
        improved_prompt += "- Questions that don't contain explicit praise\n"
        improved_prompt += "- Statements about future expectations without current quarter praise\n"
        improved_prompt += "- Technical discussions without performance praise\n"
        improved_prompt += "- Personal congratulations unrelated to company performance\n\n"
    
    if false_negative_phrases:
        improved_prompt += "IMPORTANT - DO classify these as compliments:\n"
        improved_prompt += "- Explicit congratulations on results ('Great quarter', 'Strong results')\n"
        improved_prompt += "- Praise for execution ('Great to see the execution')\n"
        improved_prompt += "- Positive feedback on performance ('Very good job in the quarter')\n"
        improved_prompt += "- Any statement containing 'congrats' or 'congratulations' about performance\n\n"
    
    improved_prompt += "REMEMBER: Only classify as compliments if the analyst is explicitly praising the company's performance or results for the current quarter.\n"
    
    return improved_prompt

def main():
    target_tickers = ['ADMA']

    # First iteration with existing prompt
    print("="*80)
    print("ITERATION 1: Using original prompt")
    print("="*80)
    
    run_name = 'V0'
    results_path = os.path.join('results', f"results_{run_name}")
    evaluation_path = os.path.join('evaluation', f"evaluation_{run_name}")

    print(f"results_path = {results_path}")
    print(f"evaluation_path = {evaluation_path}")
    
    # Run detection
    prompt_detection_path = 'prompts/detection_prompt_nominal.txt'
    results_path = run_detection_only(prompt_detection_path=prompt_detection_path, target_tickers=target_tickers, results_path=results_path)
    
    # Run evaluation on this result
    score_v0, results_df_v0 = run_evaluation(tickers=target_tickers, resdir=results_path, outputdir=evaluation_path)
    print(f"\nV0 Score: {score_v0}")
    
    # Analyze errors from first iteration
    print("\n" + "="*80)
    print("ANALYZING ERRORS FROM FIRST ITERATION")
    print("="*80)
    
    error_analysis = analyze_errors(evaluation_path, target_tickers)
    print_error_analysis(error_analysis)
    
    # Create improved prompt
    print("\n" + "="*80)
    print("CREATING IMPROVED PROMPT")
    print("="*80)
    
    improved_prompt = create_improved_prompt(prompt_detection_path, error_analysis)
    
    # Save improved prompt
    improved_prompt_path = 'prompts/detection_prompt_improved.txt'
    with open(improved_prompt_path, 'w', encoding='utf-8') as f:
        f.write(improved_prompt)
    
    print(f"Improved prompt saved to: {improved_prompt_path}")
    
    # Second iteration with improved prompt
    print("\n" + "="*80)
    print("ITERATION 2: Using improved prompt")
    print("="*80)
    
    run_name_v1 = 'V1'
    results_path_v1 = os.path.join('results', f"results_{run_name_v1}")
    evaluation_path_v1 = os.path.join('evaluation', f"evaluation_{run_name_v1}")

    print(f"results_path_v1 = {results_path_v1}")
    print(f"evaluation_path_v1 = {evaluation_path_v1}")
    
    # Run detection with improved prompt
    results_path_v1 = run_detection_only(prompt_detection_path=improved_prompt_path, target_tickers=target_tickers, results_path=results_path_v1)
    
    # Run evaluation on improved results
    score_v1, results_df_v1 = run_evaluation(tickers=target_tickers, resdir=results_path_v1, outputdir=evaluation_path_v1)
    print(f"\nV1 Score: {score_v1}")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON OF RESULTS")
    print("="*80)
    print(f"V0 Score: {score_v0:.4f}")
    print(f"V1 Score: {score_v1:.4f}")
    print(f"Improvement: {score_v1 - score_v0:.4f}")
    
    if score_v1 > score_v0:
        print("✓ V1 performed better than V0!")
    else:
        print("✗ V1 did not improve over V0")
    
    # Print detailed comparison
    print("\nDetailed Metrics Comparison:")
    print("V0 Metrics:")
    print(results_df_v0[['ticker', 'accuracy', 'precision', 'recall']])
    print("\nV1 Metrics:")
    print(results_df_v1[['ticker', 'accuracy', 'precision', 'recall']])

if __name__ == "__main__":
    main()

