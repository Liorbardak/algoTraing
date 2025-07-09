import json
import sys
from collections import defaultdict

# Usage: python compare_compliments.py actual.json gt.json before_validation.json output.json

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def index_compliments_by_name(compliments):
    # Normalize names for matching (strip, lower)
    return {c['analyst_name'].strip().lower(): c for c in compliments}

def names_match(name1, name2):
    n1 = name1.strip().lower()
    n2 = name2.strip().lower()
    return n1 == n2 or n1 in n2 or n2 in n1

def find_matching_before_validation(analyst_name, before_validation_comps):
    for c in before_validation_comps:
        if names_match(analyst_name, c['analyst_name']):
            return c
    return None

def compare_analyst_compliments(actual, gt, before_validation):
    results = []
    stats = defaultdict(int)
    all_quarters = set(actual.keys()) & set(gt.keys())
    for quarter in all_quarters:
        actual_comps = actual[quarter]['compliments']
        gt_comps = gt[quarter]['compliments']
        before_validation_comps = before_validation.get(quarter, {}).get('compliments', [])
        actual_by_name = index_compliments_by_name(actual_comps)
        gt_by_name = index_compliments_by_name(gt_comps)
        all_names = set(actual_by_name.keys()) | set(gt_by_name.keys())
        matched_gt = set()
        matched_actual = set()
        # First, try exact and partial matches
        for actual_name, actual_c in actual_by_name.items():
            found = False
            for gt_name, gt_c in gt_by_name.items():
                if names_match(actual_c['analyst_name'], gt_c['analyst_name']):
                    found = True
                    matched_gt.add(gt_name)
                    matched_actual.add(actual_name)
                    before_c = find_matching_before_validation(actual_c['analyst_name'], before_validation_comps)
                    before_name = before_c['analyst_name'] if before_c else None
                    before_quoted = before_c['quoted_compliment'] if before_c else None
                    if actual_c['level'] == gt_c['level']:
                        stats['correct'] += 1
                    else:
                        stats['incorrect_level'] += 1
                    # Always output if not exact match
                    partial = actual_c['analyst_name'].strip().lower() != gt_c['analyst_name'].strip().lower()
                    if partial:
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
                    elif actual_c['level'] != gt_c['level']:
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
                # No match found in GT for this actual
                before_c = find_matching_before_validation(actual_c['analyst_name'], before_validation_comps)
                before_name = before_c['analyst_name'] if before_c else None
                before_quoted = before_c['quoted_compliment'] if before_c else None
                stats['incorrect_name'] += 1
                results.append({
                    'quarter': quarter,
                    'analyst_name': actual_c['analyst_name'],
                    'analyst_name_before_validation': before_name,
                    'analyst_name_GT': None,
                    'level': actual_c['level'],
                    'level_GT': None,
                    'quoted_compliment': actual_c['quoted_compliment'],
                    'quoted_compliment_before_validation': before_quoted,
                    'quoted_compliment_GT': None,
                    'partial_name_match': False
                })
        # Now, check for GT names not matched by actual
        for gt_name, gt_c in gt_by_name.items():
            if gt_name not in matched_gt:
                before_c = find_matching_before_validation(gt_c['analyst_name'], before_validation_comps)
                before_name = before_c['analyst_name'] if before_c else None
                before_quoted = before_c['quoted_compliment'] if before_c else None
                stats['incorrect_name'] += 1
                results.append({
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
    return results, stats

def main():
    # if len(sys.argv) != 4:
    #     print('Usage: python compare_compliments.py actual.json gt.json before_validation.json output.json')
    #     sys.exit(1)
    # actual_path, gt_path, before_validation_path, output_path = sys.argv[1:5]
    ticker = 'CLBT'
    actual_path = f'C:/Users/dadab/projects/algotrading/data/results_20250709_101434/{ticker}_all_validated_compliments.json'
    gt_path = f'C:/Users/dadab/projects/algotrading/data/GT/{ticker}_all_validated_compliments_4.7_GT.json'
    before_validation_path = f'C:/Users/dadab/projects/algotrading/data/results_20250709_101434/{ticker}_detected_compliments_before_validation.json'
    output_path = f'C:/Users/dadab/projects/algotrading/data/results_20250709_101434/{ticker}_compareToGT.json'
    actual = load_json(actual_path)
    gt = load_json(gt_path)
    before_validation = load_json(before_validation_path)
    results, stats = compare_analyst_compliments(actual, gt, before_validation)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Comparison complete. Results written to: {output_path}')
    print(f"Statistics: Correct: {stats['correct']}, Incorrect Level: {stats['incorrect_level']}, Incorrect Name: {stats['incorrect_name']}")

if __name__ == '__main__':
    main() 