import json
import sys
from collections import defaultdict
import re
import os
import pandas as pd


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_dual_field_structure(data, include_quote3_in_results=True):
    """
    Normalize data to legacy structure.
    Automatically detects and handles both legacy and new three-quote structures.
    For three-quote structure:
        - level = level_quote1 OR level_quote2 OR level_quote3 (if include_quote3_in_results=True)
        - level = level_quote1 OR level_quote2 (if include_quote3_in_results=False)
    For legacy structure: keeps as-is
    
    Args:
        data: Data with either legacy or three-quote structure
        include_quote3_in_results: Whether to include quote3 level in results calculation
    """
    normalized_data = {}
    
    for quarter_key, quarter_data in data.items():
        normalized_quarter = {
            'year': quarter_data.get('year'),
            'quarter': quarter_data.get('quarter'),
            'date': quarter_data.get('date'),
            'compliments': []
        }
        
        compliments = quarter_data.get('compliments', [])
        for compliment in compliments:
            # Check if this is three-quote structure
            has_level_quote1 = 'level_quote1' in compliment
            has_level_quote2 = 'level_quote2' in compliment
            has_level_quote3 = 'level_quote3' in compliment
            
            if has_level_quote1 and has_level_quote2 and has_level_quote3:
                # Three-quote structure - normalize to legacy
                level_quote1 = compliment.get('level_quote1', 0)
                level_quote2 = compliment.get('level_quote2', 0)
                level_quote3 = compliment.get('level_quote3', 0)
                
                # Calculate level based on parameter
                if include_quote3_in_results:
                    # Option 1: level_quote1 OR level_quote2 OR level_quote3
                    level = 1 if (level_quote1 == 1 or level_quote2 == 1 or level_quote3 == 1) else 0
                else:
                    # Option 2: level_quote1 OR level_quote2 (ignore quote3)
                    level = 1 if (level_quote1 == 1 or level_quote2 == 1) else 0
                
                # quoted_compliment = concatenation of all non-empty quotes
                quote1 = compliment.get('quote1', '').strip()
                quote2 = compliment.get('quote2', '').strip()
                quote3 = compliment.get('quote3', '').strip()
                
                # Collect all non-empty quotes
                quotes = []
                if quote1:
                    quotes.append(quote1)
                if quote2:
                    quotes.append(quote2)
                if quote3:
                    quotes.append(quote3)
                
                # Concatenate with spaces
                normalized_quoted_compliment = " ".join(quotes) if quotes else ""
                
                normalized_compliment = {
                    'analyst_name': compliment.get('analyst_name', ''),
                    'level': level,
                    'quoted_compliment': normalized_quoted_compliment
                }
                
            elif 'level_compliment' in compliment and 'level_start' in compliment:
                # Old dual-field structure - normalize to legacy
                level_compliment = compliment.get('level_compliment', 0)
                level_start = compliment.get('level_start', 0)
                
                # level = level_compliment OR level_start
                level = 1 if (level_compliment == 1 or level_start == 1) else 0
                
                # quoted_compliment = concatenation of quoted_compliment and comment_start
                quoted_compliment = compliment.get('quoted_compliment', '')
                comment_start = compliment.get('comment_start', '')
                
                # Concatenate with space if both are non-empty
                if quoted_compliment and comment_start:
                    normalized_quoted_compliment = f"{quoted_compliment} {comment_start}"
                elif quoted_compliment:
                    normalized_quoted_compliment = quoted_compliment
                elif comment_start:
                    normalized_quoted_compliment = comment_start
                else:
                    normalized_quoted_compliment = ""
                
                normalized_compliment = {
                    'analyst_name': compliment.get('analyst_name', ''),
                    'level': level,
                    'quoted_compliment': normalized_quoted_compliment
                }
                
            elif 'level' in compliment:
                # Already legacy structure - keep as is
                normalized_compliment = {
                    'analyst_name': compliment.get('analyst_name', ''),
                    'level': compliment.get('level', 0),
                    'quoted_compliment': compliment.get('quoted_compliment', '')
                }
            else:
                # Unknown structure - assume no compliment
                normalized_compliment = {
                    'analyst_name': compliment.get('analyst_name', ''),
                    'level': 0,
                    'quoted_compliment': ''
                }
            
            normalized_quarter['compliments'].append(normalized_compliment)
        
        normalized_data[quarter_key] = normalized_quarter
    
    return normalized_data


def normalize_gt_structure(data):
    """
    Normalize ground truth data to legacy structure.
    Automatically detects and handles both legacy and new three-quote structures.
    For legacy structure: keeps as-is
    For three-quote structure: converts to legacy with quote3 always included
    """
    if not data:
        return data
    
    # Determine structure type by checking the first compliment in the first quarter
    first_quarter = list(data.keys())[0] if data else None
    if not first_quarter or 'compliments' not in data[first_quarter] or not data[first_quarter]['compliments']:
        return data
    
    first_compliment = data[first_quarter]['compliments'][0]
    
    # Check if this is three-quote structure
    if 'level_quote1' in first_compliment and 'level_quote2' in first_compliment and 'level_quote3' in first_compliment:
        # Three-quote structure - normalize to legacy
        print(f"   Converting three-quote GT structure to legacy structure")
        return _normalize_three_quote_to_legacy(data)
    elif 'level' in first_compliment:
        # Already legacy structure - return as-is
        print(f"   GT already in legacy structure - keeping as-is")
        return data
    else:
        # Unknown structure - return as-is
        print(f"   Unknown GT structure - keeping as-is")
        return data


def _normalize_three_quote_to_legacy(data):
    """Helper function to convert three-quote structure to legacy structure"""
    print(f"      Converting {len(data)} quarters from three-quote to legacy structure")
    normalized_data = {}
    
    for quarter_key, quarter_data in data.items():
        normalized_quarter = {
            'year': quarter_data.get('year'),
            'quarter': quarter_data.get('quarter'),
            'date': quarter_data.get('date'),
            'compliments': []
        }
        
        compliments = quarter_data.get('compliments', [])
        print(f"      Processing {len(compliments)} compliments in {quarter_key}")
        
        for compliment in compliments:
            # Extract three-quote data
            level_quote1 = compliment.get('level_quote1', 0)
            level_quote2 = compliment.get('level_quote2', 0)
            level_quote3 = compliment.get('level_quote3', 0)
            
            # Ground truth ALWAYS includes quote3: level_quote1 OR level_quote2 OR level_quote3
            level = 1 if (level_quote1 == 1 or level_quote2 == 1 or level_quote3 == 1) else 0
            
            # Combine all quotes
            quote1 = compliment.get('quote1', '').strip()
            quote2 = compliment.get('quote2', '').strip()
            quote3 = compliment.get('quote3', '').strip()
            
            quotes = []
            if quote1:
                quotes.append(quote1)
            if quote2:
                quotes.append(quote2)
            if quote3:
                quotes.append(quote3)
            
            quoted_compliment = " ".join(quotes) if quotes else ""
            
            normalized_compliment = {
                'analyst_name': compliment.get('analyst_name', ''),
                'level': level,
                'quoted_compliment': quoted_compliment
            }
            
            normalized_quarter['compliments'].append(normalized_compliment)
        
        normalized_data[quarter_key] = normalized_quarter
    
    print(f"      Conversion complete: {len(normalized_data)} quarters normalized")
    return normalized_data


def index_compliments_by_name(compliments):
    # Normalize names for matching (strip, lower)
    return {c['analyst_name'].strip().lower(): c for c in compliments}


def names_match(name1, name2):
    n1 = name1.strip().lower()
    n2 = name2.strip().lower()
    return n1 == n2 or n1 in n2 or n2 in n1


def find_matching_before_validation(analyst_name, before_validation_comps):
    for c in before_validation_comps:
        if analyst_name == c['analyst_name']:
            return c
    return None


def split_words(name):
    # Remove parentheses and split into words, lowercased
    name = re.sub(r'[()\[\]]', ' ', name)
    return set(w.lower() for w in re.split(r'\W+', name) if w)


def best_name_match(actual_name, gt_by_name):
    actual_words = split_words(actual_name)
    best_match = None
    best_score = 0
    second_best_score = 0
    for gt_name, gt_c in gt_by_name.items():
        gt_words = split_words(gt_c['analyst_name'])
        overlap = len(actual_words & gt_words)
        if overlap > best_score:
            second_best_score = best_score
            best_score = overlap
            best_match = gt_name
        elif overlap > second_best_score:
            second_best_score = overlap
    # Only match if at least 2 words overlap and best_score - second_best_score >= 1
    if best_score >= 2 and (best_score - second_best_score) >= 1:
        return best_match
    return None


def compare_analyst_compliments(actual, gt, before_validation):
    results = []
    results_incorrect_names = []
    stats = defaultdict(int)
    all_quarters = sorted(list(set(actual.keys()) & set(gt.keys())))
    for quarter in all_quarters:
        actual_comps = actual[quarter]['compliments']
        gt_comps = gt[quarter]['compliments']
        before_validation_comps = before_validation.get(quarter, {}).get('compliments', [])
        actual_by_name = index_compliments_by_name(actual_comps)
        gt_by_name = index_compliments_by_name(gt_comps)
        matched_gt = set()
        matched_actual = set()
        stats['n_gt_analyst'] += len(gt_comps)
        stats['n_actual_analyst'] += len(actual_comps)
        # First, try exact and advanced matches
        for actual_name, actual_c in actual_by_name.items():
            found = False
            # 1. Try exact match (case-insensitive)
            for gt_name, gt_c in gt_by_name.items():
                if actual_c['analyst_name'].strip().lower() == gt_c['analyst_name'].strip().lower():
                    found = True
                    matched_gt.add(gt_name)
                    matched_actual.add(actual_name)
                    before_c = find_matching_before_validation(actual_c['analyst_name'], before_validation_comps)
                    before_name = before_c['analyst_name'] if before_c else None
                    try:
                        before_quoted = before_c['quoted_compliment'] if before_c else None
                    except:
                        before_quoted = "read error"
                    
                    # Check if gt_c has the expected structure
                    if 'level' not in gt_c:
                        print(f"❌ GT entry missing 'level' field: {list(gt_c.keys())}")
                        print(f"   Quarter: {quarter}, Analyst: {gt_c.get('analyst_name', 'Unknown')}")
                        continue
                        
                    if gt_c['level']:
                        stats['n_positive'] += 1
                    else:
                        stats['n_negative'] += 1
                    if actual_c['level'] == gt_c['level']:
                        stats['correct'] += 1
                    else:
                        stats['incorrect_level'] += 1
                        if actual_c['level'] == 0:
                            stats['false_negative'] += 1
                        else:
                            stats['false_positive'] += 1
                        results.append({
                            'quarter': quarter,
                            'analyst_name': actual_c['analyst_name'],
                            'analyst_name_before_validation': before_name,
                            'analyst_name_GT': gt_c['analyst_name'],
                            'level': actual_c['level'],
                            'level_GT': gt_c['level'],
                            'quoted_compliment': actual_c['quoted_compliment'],
                            'quoted_compliment_before_validation': before_quoted,
                            'quoted_compliment_GT': gt_c['quoted_compliment'],
                            'partial_name_match': False
                        })
                    break
            if not found:
                # 2. Try partial name match
                best_match = best_name_match(actual_c['analyst_name'], gt_by_name)
                if best_match and best_match not in matched_gt:
                    found = True
                    matched_gt.add(best_match)
                    matched_actual.add(actual_name)
                    gt_c = gt_by_name[best_match]
                    before_c = find_matching_before_validation(actual_c['analyst_name'], before_validation_comps)
                    before_name = before_c['analyst_name'] if before_c else None
                    try:
                        before_quoted = before_c['quoted_compliment'] if before_c else None
                    except:
                        before_quoted = "read error"
                    
                    # Check if gt_c has the expected structure
                    if 'level' not in gt_c:
                        print(f"❌ GT entry missing 'level' field: {list(gt_c.keys())}")
                        print(f"   Quarter: {quarter}, Analyst: {gt_c.get('analyst_name', 'Unknown')}")
                        continue
                        
                    if gt_c['level']:
                        stats['n_positive'] += 1
                    else:
                        stats['n_negative'] += 1
                    if actual_c['level'] == gt_c['level']:
                        stats['correct'] += 1
                    else:
                        stats['incorrect_level'] += 1
                        if actual_c['level'] == 0:
                            stats['false_negative'] += 1
                        else:
                            stats['false_positive'] += 1
                        results.append({
                            'quarter': quarter,
                            'analyst_name': actual_c['analyst_name'],
                            'analyst_name_before_validation': before_name,
                            'analyst_name_GT': gt_c['analyst_name'],
                            'level': actual_c['level'],
                            'level_GT': gt_c['level'],
                            'quoted_compliment': actual_c['quoted_compliment'],
                            'quoted_compliment_before_validation': before_quoted,
                            'quoted_compliment_GT': gt_c['quoted_compliment'],
                            'partial_name_match': True
                        })
            if not found:
                # 3. No match found - count as error
                stats['incorrect_name'] += 1
                results_incorrect_names.append({
                    'quarter': quarter,
                    'analyst_name': actual_c['analyst_name'],
                    'analyst_name_before_validation': actual_c['analyst_name'],
                    'analyst_name_GT': None,
                    'level': actual_c['level'],
                    'level_GT': None,
                    'quoted_compliment': actual_c['quoted_compliment'],
                    'quoted_compliment_before_validation': actual_c['quoted_compliment'],
                    'quoted_compliment_GT': None,
                    'partial_name_match': False
                })

        # Now, check for GT names not matched by actual
        for gt_name, gt_c in gt_by_name.items():
            if gt_name not in matched_gt:
                before_c = find_matching_before_validation(gt_c['analyst_name'], before_validation_comps)
                before_name = before_c['analyst_name'] if before_c else None
                before_quoted = before_c['quoted_compliment'] if before_c else None
                
                # Check if gt_c has the expected structure
                if 'level' not in gt_c:
                    print(f"❌ GT entry missing 'level' field: {list(gt_c.keys())}")
                    print(f"   Quarter: {quarter}, Analyst: {gt_c.get('analyst_name', 'Unknown')}")
                    continue
                    
                if gt_c['level'] == 0:
                    # Do not count as error if GT level is 0
                    continue
                stats['incorrect_name'] += 1
                results_incorrect_names.append({
                    'quarter': quarter,
                    'analyst_name': None,
                    'analyst_name_before_validation': before_name,
                    'analyst_name_GT': gt_c['analyst_name'],
                    'level': None,
                    'level_GT': gt_c['level'],
                    'quoted_compliment': None,
                    'quoted_compliment_before_validation': before_quoted,
                    'quoted_compliment_GT': gt_c['quoted_compliment'],
                    'partial_name_match': False
                })
    return results, results_incorrect_names, stats


def main():
    # Configuration
    include_quote3_in_results = True  # Set this to False if you don't want to include quote3 in results
    
    # Set paths
    basepath = '../../../../data/'
    resdir = os.path.join(basepath, "results/allresults")
    gtdir = os.path.join(basepath, "results/allGT")
    outputdir = os.path.join(resdir, 'compareToGT')
    os.makedirs(outputdir, exist_ok=True)
    statistic_filename = os.path.join(outputdir, f"compareToGT_include_quote3_{include_quote3_in_results}.csv")
    # Default: process all tickers, or specify a specific list for testing
    #tickers = ["ADM"]  # Uncomment and modify this line to test specific tickers
    
    # Auto-detect all unique tickers from the results folder
    if 'tickers' not in locals() or not tickers:
        print("Auto-detecting tickers from results folder...")
        ticker_files = []
        for file in os.listdir(resdir):
            if file.endswith('.json'):
                # Extract base ticker name (before first underscore)
                filename = file.replace('.json', '')
                ticker = filename.split('_')[0] if '_' in filename else filename
                if ticker not in ticker_files:  # Avoid duplicates
                    ticker_files.append(ticker)
        
        tickers = ticker_files
        print(f"Found {len(tickers)} unique tickers: {', '.join(tickers)}")
    else:
        print(f"Using specified tickers: {', '.join(tickers)}")
    
    res = []
    for ticker in tickers:
        # Set paths
        compliments_file = os.path.join(resdir, f"{ticker}_all_validated_compliments.json")
        #compliments_file = os.path.join(resdir, f"{ticker}_all_validated_compliments.json")
        before_validation_file = os.path.join(resdir, f"{ticker}_detected_compliments_before_validation.json")
        # Look for GT files with priority: exact match first, then prefix match
        exact_matches = [f for f in os.listdir(gtdir) if f.startswith(f"{ticker}_") or f.startswith(f"{ticker}.")]
        prefix_matches = [f for f in os.listdir(gtdir) if f.startswith(ticker) and f not in exact_matches]
        
        # Prioritize exact matches over prefix matches
        gt_files = exact_matches + prefix_matches
        
        if not gt_files:
            print(f"No ground truth files found for {ticker}* in {gtdir}")
            continue
        
        # Use the first matching GT file
        gt_filename = gt_files[0]
        gt_file = os.path.join(gtdir, gt_filename)
        print(f"Using GT file: {gt_filename}")

        gt_level_errors_output_file = os.path.join(outputdir, f"{ticker}_compareToGT_level_errors_include_quote3_{include_quote3_in_results}.json")
        name_errors_output_file = os.path.join(outputdir, f"{ticker}_compareToGT_name_errors_include_quote3_{include_quote3_in_results}.json")

        # Load json
        actual = load_json(compliments_file)
            
        try:
            gt = load_json(gt_file)
        except json.JSONDecodeError as e:
            print(f"JSON decode error loading ground truth file {gt_file}: {e}")
            continue
        except Exception as e:
            print(f"Unexpected error loading ground truth file {gt_file}: {str(e)}")
            continue
            
        before_validation = load_json(before_validation_file)

        # Normalize actual data to legacy structure (with configurable quote3 inclusion)
        actual = normalize_dual_field_structure(actual, include_quote3_in_results)
        before_validation = normalize_dual_field_structure(before_validation, include_quote3_in_results)
        
        # Normalize ground truth data to legacy structure (handles both legacy and three-quote structures)
        gt = normalize_gt_structure(gt)  # Use proper GT normalization function
        
        # Debug: Check if normalization worked
        if gt and isinstance(gt, dict) and gt:
            first_quarter = list(gt.keys())[0]
            if first_quarter and 'compliments' in gt[first_quarter]:
                print(f"   After GT normalization: {len(gt[first_quarter]['compliments'])} compliments in {first_quarter}")
                if gt[first_quarter]['compliments']:
                    first_comp = gt[first_quarter]['compliments'][0]
                    print(f"   First compliment structure: {list(first_comp.keys())}")
        
        # Run compare
        results, results_incorrect_names, stats = compare_analyst_compliments(actual, gt, before_validation)

        with open(gt_level_errors_output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        with open(name_errors_output_file, 'w', encoding='utf-8') as f:
            json.dump(results_incorrect_names, f, indent=2, ensure_ascii=False)

        print(f"Statistics for {ticker}: Correct: {stats['correct']}, Incorrect Level: {stats['incorrect_level']}, Incorrect Name: {stats['incorrect_name']}")
        res.append({'ticker': ticker,
                   'n_gt_analysits': stats['n_gt_analyst'],
                   'n_actual_analyst': stats['n_actual_analyst'],
                   'n_analysts_matched': stats['correct'] + stats['incorrect_level'],
                   'correct': stats['correct'],
                   'incorrect': stats['incorrect_level'],
                   'false_positive': stats['false_positive'],
                   'false_negative': stats['false_negative'],
                   'n_positive': stats['n_positive'],
                   'n_negative': stats['n_negative'],
                   })

    df = pd.DataFrame(res)
    df.loc['Total'] = df.sum()
    df.loc['Total', 'ticker'] = 'all'
    df['accuracy'] = df['correct'] / (df['correct'] + df['incorrect'])

    TP = df['n_positive'] - df['false_negative']
    df['precision'] = TP / (TP + df['false_positive'])
    df['recall'] = (df['n_positive'] - df['false_negative']) / df['n_positive']
    df.to_csv(statistic_filename)

    # Set option to display all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Results normalization: {'Include quote3' if include_quote3_in_results else 'Exclude quote3'}")
    print(f"Ground truth normalization: Always include quote3")
    print(f"Output file: {statistic_filename}")
    print("="*60)
    
    print(df[['ticker', 'accuracy', 'precision', 'recall']])


if __name__ == '__main__':
    main() 