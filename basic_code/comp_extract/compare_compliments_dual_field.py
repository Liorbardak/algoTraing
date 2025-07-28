import os
import json
import glob
from collections import defaultdict


def load_json_file(file_path):
    """Load and return JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_analyst_compliments(data):
    """
    Extract analyst compliments from either dual-field structure or legacy structure.
    For dual-field: An analyst qualifies if either level_compliment OR level_start is 1.
    For legacy: An analyst qualifies if level is 1.
    
    Args:
        data (dict): The earnings call data with quarters and compliments
        
    Returns:
        dict: Dictionary mapping analyst names to their compliment status
    """
    analyst_compliments = {}
    
    for quarter_key, quarter_data in data.items():
        compliments = quarter_data.get('compliments', [])
        
        for compliment in compliments:
            analyst_name = compliment.get('analyst_name', '')
            if not analyst_name:
                continue
            
            # Check if this is dual-field structure or legacy structure
            has_level_compliment = 'level_compliment' in compliment
            has_level_start = 'level_start' in compliment
            has_level = 'level' in compliment
            
            if has_level_compliment and has_level_start:
                # Dual-field structure
                level_compliment = compliment.get('level_compliment', 0)
                level_start = compliment.get('level_start', 0)
                has_compliment = (level_compliment == 1) or (level_start == 1)
                
                compliment_info = {
                    'has_compliment': has_compliment,
                    'level_compliment': level_compliment,
                    'level_start': level_start,
                    'quoted_compliment': compliment.get('quoted_compliment', ''),
                    'comment_start': compliment.get('comment_start', ''),
                    'structure': 'dual_field'
                }
            elif has_level:
                # Legacy structure
                level = compliment.get('level', 0)
                has_compliment = (level == 1)
                
                compliment_info = {
                    'has_compliment': has_compliment,
                    'level': level,
                    'quoted_compliment': compliment.get('quoted_compliment', ''),
                    'structure': 'legacy'
                }
            else:
                # Unknown structure, assume no compliment
                has_compliment = False
                compliment_info = {
                    'has_compliment': has_compliment,
                    'structure': 'unknown'
                }
            
            # Store the compliment status for this analyst
            if analyst_name not in analyst_compliments:
                analyst_compliments[analyst_name] = compliment_info
            else:
                # If analyst appears in multiple quarters, update if they have a compliment
                if has_compliment:
                    analyst_compliments[analyst_name] = compliment_info
    
    return analyst_compliments


def compare_compliments_for_ticker(ticker, resdir, gtdir):
    """
    Compare detected compliments with ground truth for a specific ticker.
    
    Args:
        ticker (str): The ticker symbol
        resdir (str): Directory containing results
        gtdir (str): Directory containing ground truth files
        
    Returns:
        dict: Comparison results
    """
    print(f"\n{'='*60}")
    print(f"Processing Ticker: {ticker}")
    print(f"{'='*60}")
    
    # Find the result file
    result_file = os.path.join(resdir, f"{ticker}_all_validated_compliments.json")
    if not os.path.exists(result_file):
        print(f"❌ Result file not found: {result_file}")
        return None
    
    # Find the ground truth file using wildcard
    gt_file_pattern = os.path.join(gtdir, f"{ticker}*")
    gt_files = glob.glob(gt_file_pattern)
    if not gt_files:
        print(f"❌ No ground truth file found for {ticker} with pattern {gt_file_pattern}")
        return None
    
    gt_file = gt_files[0]  # Use the first match
    print(f"📁 Using ground truth file: {os.path.basename(gt_file)}")
    
    # Load data
    detected_data = load_json_file(result_file)
    gt_data = load_json_file(gt_file)
    
    if detected_data is None or gt_data is None:
        return None
    
    # Extract analyst compliments from detected data
    detected_analysts = extract_analyst_compliments(detected_data)
    
    # Extract analyst compliments from ground truth
    gt_analysts = extract_analyst_compliments(gt_data)
    
    print(f"\n📊 Analysis for {ticker}:")
    print(f"   Detected analysts: {len(detected_analysts)}")
    print(f"   Ground truth analysts: {len(gt_analysts)}")
    
    # Get all unique analyst names
    all_analysts = set(detected_analysts.keys()) | set(gt_analysts.keys())
    
    # Initialize counters
    tp = 0  # True Positives: Correctly identified as having compliment
    fp = 0  # False Positives: Incorrectly identified as having compliment
    fn = 0  # False Negatives: Missed analysts with compliments
    tn = 0  # True Negatives: Correctly identified as not having compliment
    
    # Detailed comparison
    comparison_details = {
        'true_positives': [],
        'false_positives': [],
        'false_negatives': [],
        'true_negatives': []
    }
    
    for analyst in all_analysts:
        detected_has_compliment = detected_analysts.get(analyst, {}).get('has_compliment', False)
        gt_has_compliment = gt_analysts.get(analyst, {}).get('has_compliment', False)
        
        if detected_has_compliment and gt_has_compliment:
            # True Positive
            tp += 1
            comparison_details['true_positives'].append({
                'analyst': analyst,
                'detected': detected_analysts.get(analyst, {}),
                'ground_truth': gt_analysts.get(analyst, {})
            })
        elif detected_has_compliment and not gt_has_compliment:
            # False Positive
            fp += 1
            comparison_details['false_positives'].append({
                'analyst': analyst,
                'detected': detected_analysts.get(analyst, {}),
                'ground_truth': gt_analysts.get(analyst, {})
            })
        elif not detected_has_compliment and gt_has_compliment:
            # False Negative
            fn += 1
            comparison_details['false_negatives'].append({
                'analyst': analyst,
                'detected': detected_analysts.get(analyst, {}),
                'ground_truth': gt_analysts.get(analyst, {})
            })
        else:
            # True Negative
            tn += 1
            comparison_details['true_negatives'].append({
                'analyst': analyst,
                'detected': detected_analysts.get(analyst, {}),
                'ground_truth': gt_analysts.get(analyst, {})
            })
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    results = {
        'ticker': ticker,
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        },
        'counts': {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'true_negatives': tn,
            'total_analysts': len(all_analysts)
        },
        'comparison_details': comparison_details
    }
    
    # Print results
    print(f"\n📈 Results for {ticker}:")
    print(f"   Precision: {precision:.3f} ({tp}/{tp + fp})")
    print(f"   Recall: {recall:.3f} ({tp}/{tp + fn})")
    print(f"   F1-Score: {f1_score:.3f}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   True Positives: {tp}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   True Negatives: {tn}")
    
    # Print detailed breakdown
    if comparison_details['false_positives']:
        print(f"\n❌ False Positives (detected but not in ground truth):")
        for item in comparison_details['false_positives']:
            analyst = item['analyst']
            detected = item['detected']
            structure = detected.get('structure', 'unknown')
            if structure == 'dual_field':
                print(f"   - {analyst}: level_compliment={detected.get('level_compliment')}, level_start={detected.get('level_start')}")
            elif structure == 'legacy':
                print(f"   - {analyst}: level={detected.get('level')}")
            else:
                print(f"   - {analyst}: structure={structure}")
    
    if comparison_details['false_negatives']:
        print(f"\n❌ False Negatives (missed analysts with compliments):")
        for item in comparison_details['false_negatives']:
            analyst = item['analyst']
            gt = item['ground_truth']
            structure = gt.get('structure', 'unknown')
            if structure == 'dual_field':
                print(f"   - {analyst}: level_compliment={gt.get('level_compliment')}, level_start={gt.get('level_start')}")
            elif structure == 'legacy':
                print(f"   - {analyst}: level={gt.get('level')}")
            else:
                print(f"   - {analyst}: structure={structure}")
    
    return results


def main():
    """
    Main function to compare compliments across multiple tickers.
    """
    # Configuration - update these paths as needed
    resdir = "../../../../data/results/results_dual_field_20250725_114951/"
    gtdir = "../../../../data/results/GT/"
    
    # List of tickers to process
    tickers = ["ADMA"]  # Add more tickers as needed
    
    print("🔍 Starting Dual-Field Compliment Comparison")
    print(f"📁 Results directory: {resdir}")
    print(f"📁 Ground truth directory: {gtdir}")
    
    all_results = []
    
    for ticker in tickers:
        result = compare_compliments_for_ticker(ticker, resdir, gtdir)
        if result:
            all_results.append(result)
    
    # Calculate overall metrics
    if all_results:
        total_tp = sum(r['counts']['true_positives'] for r in all_results)
        total_fp = sum(r['counts']['false_positives'] for r in all_results)
        total_fn = sum(r['counts']['false_negatives'] for r in all_results)
        total_tn = sum(r['counts']['true_negatives'] for r in all_results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0
        
        print(f"\n{'='*80}")
        print("📊 OVERALL RESULTS")
        print(f"{'='*80}")
        print(f"Overall Precision: {overall_precision:.3f}")
        print(f"Overall Recall: {overall_recall:.3f}")
        print(f"Overall F1-Score: {overall_f1:.3f}")
        print(f"Overall Accuracy: {overall_accuracy:.3f}")
        print(f"Total True Positives: {total_tp}")
        print(f"Total False Positives: {total_fp}")
        print(f"Total False Negatives: {total_fn}")
        print(f"Total True Negatives: {total_tn}")
    
    print(f"\n✅ Comparison completed for {len(all_results)} tickers")


if __name__ == "__main__":
    main() 