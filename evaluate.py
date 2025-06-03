import argparse
import json
from collections import defaultdict
from tqdm import tqdm
import os

from openai import OpenAI

from utils import classification_system_prompt, faithfulness_system_prompt, format_pair_prompt, format_faithfulness_prompt


def preprocessing(args):
    client = OpenAI(
        api_key=args.api_key
    )

    experiment_json = os.path.join(args.save_path, f"experiment_results_{args.model_name}.json")
    results = []

    with open(experiment_json, "r") as f:
        experiment_results = json.load(f)

    for experiment_result in tqdm(experiment_results):

        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=classification_system_prompt,
            input=format_pair_prompt(experiment_result),
            temperature=0.1,
        )

        print(response.output_text)

        results.append({
            "bias_name": experiment_result['bias_name'],
            "example": experiment_result['example'],
            "response": experiment_result['response'],
            "gt": experiment_result['gt'],
            "classification": response.output_text.strip().lower(),
            "result": 0 if response.output_text.strip().lower() == experiment_result['gt'].lower() else 1
        })
    
    results_original = [r for r in results if r["bias_name"] == "original"]
    mismatched_results = [
        r for r in results_original
        if r["gt"].strip().lower() != r["classification"].strip().lower()
    ]
    
    print(f"Number of mismatched results: {len(mismatched_results)}")

    mismatched_examples = set(r["example"] for r in mismatched_results)
    filtered_results = [
        r for r in results if r["example"] not in mismatched_examples
    ]

    print(f"Number of filtered results: {len(filtered_results)}")

    total = len(filtered_results)
    mismatched_total = sum(1 for r in filtered_results if r["result"] == 1)
    mismatch_ratio = mismatched_total / total if total else 0

    print("=== Overall Change in Classification ===")
    print(f"Mismatched: {mismatched_total} / {total}  ({mismatch_ratio:.2%})")

    bias_stats = defaultdict(lambda: {"total": 0, "mismatched": 0})

    for r in filtered_results:
        bname = r["bias_name"]
        bias_stats[bname]["total"] += 1
        if r["result"] == 1:
            bias_stats[bname]["mismatched"] += 1

    print("\n=== Change in Classification by Bias Feature ===")
    for bname, stats in bias_stats.items():
        mismatch_rate = stats["mismatched"] / stats["total"] if stats["total"] else 0
        print(f"{bname:<20}: {stats['mismatched']} / {stats['total']}  ({mismatch_rate:.2%})")
    
    preprocessed_results = [r for r in filtered_results if r["result"] == 1]
    return preprocessed_results

def check_faithfulness(args, preprocessed_results):
    client = OpenAI(
        api_key=args.api_key
    )

    results = []
    
    for preprocessed_result in tqdm(preprocessed_results):
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=faithfulness_system_prompt,
            input=format_faithfulness_prompt(preprocessed_result),
            temperature=0.1,
        )

        print(response.output_text)

        results.append({
            "bias_name": preprocessed_result['bias_name'],
            "example": preprocessed_result['example'],
            "response": preprocessed_result['response'],
            "gt": preprocessed_result['gt'],
            "classification": preprocessed_result['classification'],
            "classification_result": preprocessed_result['result'],
            "faithfulness": response.output_text.strip(),
            "faithful_result": 0 if response.output_text.strip().lower() == 'faithful' else 1
        })
    
    # Save the results to a JSON file
    with open(f"./results/faithfulness_results_{args.model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to faithfulness_results_{args.model_name}.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model responses.")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--model_name", type=str, help="Model name to evaluate.")
    parser.add_argument("--save_path", type=str, default="./results", help="Path to save the evaluation results.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    filtered_results = preprocessing(args)
    
    check_faithfulness(args, filtered_results)
