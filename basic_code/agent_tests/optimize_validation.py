import os
import json
import pandas as pd
from run_validation_only import run_validation_only
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
            "You may add explicit examples for complements that the were missed , but do not add all the  false negative phrases  as examples - try to generalize"
            "Return only the improved prompt."
        )

        user_prompt = (
            f"ORIGINAL PROMPT (from previous stage):\n{original_prompt}\n\n"        
            f"FALSE NEGATIVE PHRASES (from previous stage, should be classified as compliments):\n{json.dumps(false_negative_phrases[:60], ensure_ascii=False)}\n\n"
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

def run_iteration(run_name, prompt_path, target_tickers, detection_results_path,  iteration=1):
    results_path = os.path.join('validation_results', f"results_{run_name}")
    evaluation_path = os.path.join('validation_evaluation', f"evaluation_{run_name}")
    print(f"validation_results_path = {results_path}")
    print(f"validation_evaluation_path = {evaluation_path}")
    results_path = run_validation_only(prompt_validation_path=prompt_path, target_tickers=target_tickers, detection_results_path = detection_results_path, results_path=results_path)
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
    mode = 'all'
    detection_results_path = 'detection_results/results_V1'
    # V0: Original prompt
    print("="*80)
    print("ITERATION V0: Using original prompt")
    print("="*80)
    prompt_detection_path = 'prompts/validation_prompt_nominal.txt'
    prompt_paths.append(prompt_detection_path)
    score, results_df, all_fp, all_fn = run_iteration('V0', prompt_detection_path, target_tickers,detection_results_path=detection_results_path, iteration=1)
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
    improved_prompt_path_v1 = 'prompts/validation_prompt_v1.txt'
    with open(improved_prompt_path_v1, 'w', encoding='utf-8') as f:
        f.write(improved_prompt_v1)
    prompt_paths.append(improved_prompt_path_v1)
    score, results_df, all_fp, all_fn = run_iteration('V1', improved_prompt_path_v1, target_tickers,detection_results_path=detection_results_path, iteration=1)
    scores.append(score)
    results_dfs.append(results_df)
    fp_list.append(all_fp)
    fn_list.append(all_fn)


if __name__ == "__main__":
    main()
