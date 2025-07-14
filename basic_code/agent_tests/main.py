import os
import json
import pandas as pd
from detection_only import run_detection_only
from compare_compliments import run_evaluation
from openai import OpenAI

def analyze_errors(false_positive_phrases, false_negative_phrases):
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    print(f"False Positives (should NOT be classified as compliments): {len(false_positive_phrases)}")
    for phrase in false_positive_phrases[:5]:
        print(f"  FP: '{phrase[:100]}...'")
    if len(false_positive_phrases) > 5:
        print(f"  ...and {len(false_positive_phrases)-5} more")
    print(f"False Negatives (should be classified as compliments): {len(false_negative_phrases)}")
    for phrase in false_negative_phrases[:5]:
        print(f"  FN: '{phrase[:100]}...'")
    if len(false_negative_phrases) > 5:
        print(f"  ...and {len(false_negative_phrases)-5} more")
    print()

def call_llm_to_improve_prompt(original_prompt, false_positive_phrases, false_negative_phrases, iteration=1, model="gpt-4" ,mode = 'all'):
    client = OpenAI()



    encouragement = ""
    if iteration == 2:
        encouragement = ("In this second revision, be more aggressive in changing the prompt. "
                         "Don't be afraid to restructure, add new rules, or provide more examples. ")
    elif iteration >= 3:
        encouragement = ("This is the third or later revision. Please make bold and creative changes to the prompt. "
                         "You can rewrite, reorganize, or add detailed instructions and examples. The goal is to significantly reduce the errors. ")
    if mode == 'all':
        system_prompt = (
            "You are an expert prompt engineer. "
            "Given the original prompt and lists of false positive and false negative phrases from the previous stage, "
            "revise the prompt to reduce these errors. "
            "Add explicit instructions or examples to help the LLM avoid the false positives and the false negatives errors. "
            "Return only the improved prompt."
        )

        user_prompt = (
            f"ORIGINAL PROMPT (from previous stage):\n{original_prompt}\n\n"
            f"FALSE POSITIVE PHRASES (from previous stage, should NOT be classified as compliments):\n{json.dumps(false_positive_phrases[:10], ensure_ascii=False)}\n\n"
            f"FALSE NEGATIVE PHRASES (from previous stage, should be classified as compliments):\n{json.dumps(false_negative_phrases[:10], ensure_ascii=False)}\n\n"
            f"{encouragement}Please revise the prompt accordingly."
        )
    elif mode == 'fn_only':
        system_prompt = (
            "You are an expert prompt engineer. "
            "Given the original prompt and lists of false negative phrases errors from the previous stage. "
            "These phrases were suppose to be considered as compliments , but were overlooked by the previous  stage"            
            "revise the prompt to reduce these errors. "
            "Add explicit instructions or examples to help the LLM avoid the false negatives "
            "You may add explicit examples for complements that the were missed"
            "Return only the improved prompt."
        )

        user_prompt = (
            f"ORIGINAL PROMPT (from previous stage):\n{original_prompt}\n\n"        
            f"FALSE NEGATIVE PHRASES (from previous stage, should be classified as compliments):\n{json.dumps(false_negative_phrases[:30], ensure_ascii=False)}\n\n"
            f"{encouragement}Please revise the prompt accordingly."
        )
    else:
        user_prompt = (
            f"ORIGINAL PROMPT (from previous stage):\n{original_prompt}\n\n"
            f"FALSE POSITIVE PHRASES (from previous stage, should NOT be classified as compliments):\n{json.dumps(false_positive_phrases[:10], ensure_ascii=False)}\n\n"
            f"FALSE NEGATIVE PHRASES (from previous stage, should be classified as compliments):\n{json.dumps(false_negative_phrases[:10], ensure_ascii=False)}\n\n"
            f"{encouragement}Please revise the prompt accordingly."
        )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        max_tokens=1200
    )
    improved_prompt = response.choices[0].message.content if response.choices and response.choices[0].message and response.choices[0].message.content else ""
    return improved_prompt.strip()

def run_iteration(run_name, prompt_path, target_tickers, prev_fp=None, prev_fn=None, iteration=1):
    results_path = os.path.join('results', f"results_{run_name}")
    evaluation_path = os.path.join('evaluation', f"evaluation_{run_name}")
    print(f"results_path = {results_path}")
    print(f"evaluation_path = {evaluation_path}")
    results_path = run_detection_only(prompt_detection_path=prompt_path, target_tickers=target_tickers, results_path=results_path)
    score, results_df, all_false_positive_phrases, all_false_negative_phrases = run_evaluation(tickers=target_tickers, resdir=results_path, outputdir=evaluation_path)
    print(f"\n{run_name} Score: {score}")
    analyze_errors(all_false_positive_phrases, all_false_negative_phrases)
    return score, results_df, all_false_positive_phrases, all_false_negative_phrases

def main():
    target_tickers = ['CYBR', 'BSX']
    prompt_paths = []
    scores = []
    results_dfs = []
    fp_list = []
    fn_list = []
    mode = 'fn_only'

    # V0: Original prompt
    print("="*80)
    print("ITERATION V0: Using original prompt")
    print("="*80)
    prompt_detection_path = 'prompts/detection_prompt_nominal.txt'
    prompt_paths.append(prompt_detection_path)
    score, results_df, all_fp, all_fn = run_iteration('V0', prompt_detection_path, target_tickers, iteration=1)
    scores.append(score)
    results_dfs.append(results_df)
    fp_list.append(all_fp)
    fn_list.append(all_fn)

    # V1: LLM-improved prompt (mild)
    print("\n" + "="*80)
    print("ITERATION V1: LLM-improved prompt (mild)")
    print("="*80)
    with open(prompt_paths[-1], 'r', encoding='utf-8') as f:
        prev_prompt = f.read()
    improved_prompt_v1 = call_llm_to_improve_prompt(prev_prompt, all_fp, all_fn, iteration=1,mode=mode)
    improved_prompt_path_v1 = 'prompts/detection_prompt_v1.txt'
    with open(improved_prompt_path_v1, 'w', encoding='utf-8') as f:
        f.write(improved_prompt_v1)
    prompt_paths.append(improved_prompt_path_v1)
    score, results_df, all_fp, all_fn = run_iteration('V1', improved_prompt_path_v1, target_tickers, all_fp, all_fn, iteration=1,mode=mode)
    scores.append(score)
    results_dfs.append(results_df)
    fp_list.append(all_fp)
    fn_list.append(all_fn)

    # V2: LLM-improved prompt (aggressive)
    print("\n" + "="*80)
    print("ITERATION V2: LLM-improved prompt (aggressive)")
    print("="*80)
    with open(prompt_paths[-1], 'r', encoding='utf-8') as f:
        prev_prompt = f.read()
    improved_prompt_v2 = call_llm_to_improve_prompt(prev_prompt, all_fp, all_fn, iteration=2)
    improved_prompt_path_v2 = 'prompts/detection_prompt_v2.txt'
    with open(improved_prompt_path_v2, 'w', encoding='utf-8') as f:
        f.write(improved_prompt_v2)
    prompt_paths.append(improved_prompt_path_v2)
    score, results_df, all_fp, all_fn = run_iteration('V2', improved_prompt_path_v2, target_tickers, all_fp, all_fn, iteration=2,mode=mode)
    scores.append(score)
    results_dfs.append(results_df)
    fp_list.append(all_fp)
    fn_list.append(all_fn)

    # V3: LLM-improved prompt (very aggressive)
    print("\n" + "="*80)
    print("ITERATION V3: LLM-improved prompt (very aggressive)")
    print("="*80)
    with open(prompt_paths[-1], 'r', encoding='utf-8') as f:
        prev_prompt = f.read()
    improved_prompt_v3 = call_llm_to_improve_prompt(prev_prompt, all_fp, all_fn, iteration=3,mode=mode)
    improved_prompt_path_v3 = 'prompts/detection_prompt_v3.txt'
    with open(improved_prompt_path_v3, 'w', encoding='utf-8') as f:
        f.write(improved_prompt_v3)
    prompt_paths.append(improved_prompt_path_v3)
    score, results_df, all_fp, all_fn = run_iteration('V3', improved_prompt_path_v3, target_tickers, all_fp, all_fn, iteration=3)
    scores.append(score)
    results_dfs.append(results_df)
    fp_list.append(all_fp)
    fn_list.append(all_fn)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY OF ALL ITERATIONS")
    print("="*80)
    for i, (name, score) in enumerate(['V0', 'V1', 'V2', 'V3'], 1):
        print(f"{name} Score: {score:.4f}")
    print("\nDetailed Metrics Comparison:")
    for i, name in enumerate(['V0', 'V1', 'V2', 'V3']):
        print(f"\n{name} Metrics:")
        print(results_dfs[i][['ticker', 'accuracy', 'precision', 'recall']])

if __name__ == "__main__":
    main()

