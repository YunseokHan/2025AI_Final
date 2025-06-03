import argparse
import json
from tqdm import tqdm
import os

from datasets import Dataset
from vllm import LLM, SamplingParams

from utils import format_bias_prompt


def build_dataset(json_path):
    """
    Build a dataset from a JSON file.
    
    Args:
        json_path (str): Path to the JSON file containing the dataset.
        
    Returns:
        datasets.Dataset: A Hugging Face Dataset object.
    """
    with open(json_path, "r") as f:
        json_dataset = json.load(f)
    
    # Convert the JSON dataset to a Hugging Face Dataset
    dataset = Dataset.from_list(json_dataset['examples'])
    return dataset



def main(args):
    # Load the dataset from a JSON file
    data_path = "./sports_understanding.json"
    dataset = build_dataset(data_path)

    llm = LLM(
        model=args.model_card,
        tensor_parallel_size=1
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=40,
        max_tokens=4096,
        repetition_penalty=1.2
    )

    bias_names = ['sycophancy', 'evaluation_hacking', 'consistency', 'few-shot', 'original']

    results = []
    os.makedirs(args.save_path, exist_ok=True)
    json_path = os.path.join(args.save_path, f"experiment_results_{args.model_name}.json")

    for data in tqdm(dataset):
        example = data['input']
        gt = data['target']
        for bias_name in bias_names:
            print(f"Bias: {bias_name}, Example: {example}, GT: {gt}")
            print("Generating response...")
            conversation = format_bias_prompt(bias_name, example, gt)
            response = llm.chat(
                messages=conversation,
                sampling_params=sampling_params
            )
            answer = response[0].outputs[0].text.strip()
            print(f"Response: {answer}")

            # Store the result
            results.append({
                "bias_name": bias_name,
                "example": example,
                "gt": gt,
                "response": answer
            })
    # Save the results to a JSON file
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Experiment completed. Results saved to './results/experiment_results_{args.model_name}.json'.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with a language model.")
    parser.add_argument("--model_name", type=str, default="deepseek", help="Name of the model to use.")
    parser.add_argument("--model_card", type=str)
    parser.add_argument("--save_path", type=str, default="./results", help="Path to save the results.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.model_name == "deepseek":
        args.model_card = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    elif args.model_name == "mistral":
        args.model_card = "mistralai/Mistral-7B-Instruct-v0.3"
    elif args.model_name == "llama":
        args.model_card = "meta-llama/Llama-3.1-8B-Instruct"
    elif args.model_name == "qwen":
        args.model_card = "Qwen/Qwen3-8B"
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    main(args)
