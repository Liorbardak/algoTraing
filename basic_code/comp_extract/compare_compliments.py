import json
import sys
from collections import defaultdict
import re
import os

import pandas as pd


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
    stats = defaultdict(int)
    all_quarters = set(actual.keys()) & set(gt.keys())
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
                    before_quoted = before_c['quoted_compliment'] if before_c else None
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
            if found:
                continue
            # 2. Try advanced word overlap match
            best_gt_name = best_name_match(actual_c['analyst_name'], gt_by_name)
            if best_gt_name:
                gt_c = gt_by_name[best_gt_name]
                matched_gt.add(best_gt_name)
                matched_actual.add(actual_name)
                before_c = find_matching_before_validation(actual_c['analyst_name'], before_validation_comps)
                before_name = before_c['analyst_name'] if before_c else None
                before_quoted = before_c['quoted_compliment'] if before_c else None
                if  gt_c['level']:
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
                continue
            # No match found in GT for this actual
            before_c = find_matching_before_validation(actual_c['analyst_name'], before_validation_comps)
            before_name = before_c['analyst_name'] if before_c else None
            before_quoted = before_c['quoted_compliment'] if before_c else None
            if actual_c['level'] == 0:
                # Do not count as error if actual level is 0
                continue
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
                if gt_c['level'] == 0:
                    # Do not count as error if GT level is 0
                    continue
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

    basepath = '../../../data/'


    resdir = "tests1/results_nominal"
    #resdir = "tests1/results_3_sentences"

    #resdir = "results/results_20250709_101434"
    tickers = ['ADMA', 'ADM', 'CLBT']
    #tickers = ['ADM']
    res = []
    for ticker in tickers:
        # Set paths
        actual_path = os.path.join(basepath,f"{resdir}/{ticker}_all_validated_compliments.json")
        gt_path = os.path.join(basepath,f"GT/{ticker}_all_validated_compliments_4.7_GT.json")
        before_validation_path = os.path.join(basepath,f"{resdir}/{ticker}_detected_compliments_before_validation.json")
        output_path = os.path.join(basepath,f"{resdir}/{ticker}_compareToGT.json")

        actual = load_json(actual_path)
        gt = load_json(gt_path)
        before_validation = load_json(before_validation_path)
        results, stats = compare_analyst_compliments(actual, gt, before_validation)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
       # print(f'Comparison complete. Results written to: {output_path}')
        print(f"Statistics for {ticker}: Correct: {stats['correct']}, Incorrect Level: {stats['incorrect_level']}, Incorrect Name: {stats['incorrect_name']}")
        res.append({'ticker' : ticker,
                    'n_gt_analysits': stats['n_gt_analyst'],
                    'n_actual_analyst': stats['n_actual_analyst'],
                    'n_analysts_matched': stats['correct']+stats['incorrect_level'],
                    'correct': stats['correct'],
                    'incorrect': stats['incorrect_level'],
                    'false_positive': stats['false_positive'],
                    'false_negative': stats['false_negative'],

                    })
    df = pd.DataFrame(res)
    # Set option to display all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df)

if __name__ == '__main__':
    main()


    '''
    Statistics for ADMA: Correct: 25, Incorrect Level: 3, Incorrect Name: 0
Statistics for ADM: Correct: 132, Incorrect Level: 8, Incorrect Name: 4
Statistics for CLBT: Correct: 62, Incorrect Level: 7, Incorrect Name: 2
  ticker  n_gt_analysits  n_actual_analyst  n_analysits_matched  correct  incorrect  false_positive  false_negative
0   ADMA              29                28                   28       25          3               1               2
1    ADM             149               151                  140      132          8               5               3
2   CLBT              72                73                   69       62          7               7               0
    '''