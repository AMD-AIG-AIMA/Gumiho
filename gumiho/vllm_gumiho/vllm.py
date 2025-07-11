# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import os
os.environ["VLLM_USE_V1"] = '0'



from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


def load_prompts(dataset_path, num_prompts):
    if os.path.exists(dataset_path):
        prompts = []
        try:
            with open(dataset_path) as f:
                for line in f:
                    data = json.loads(line)
                    prompts.append(data["turns"][0])
        except Exception as e:
            print(f"Error reading dataset: {e}")
            return []
    else:
        prompts = [
            "The future of AI is"
        ]

    return prompts[:num_prompts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--max_num_seqs", type=int, default=1)
    parser.add_argument("--num_prompts", type=int, default=80)
    parser.add_argument("--num_spec_tokens", type=int, default=3)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--draft_tp", type=int, default=1)
    parser.add_argument("--enforce_eager", action='store_true')
    parser.add_argument("--enable_chunked_prefill", action='store_true')
    parser.add_argument("--max_num_batched_tokens", type=int, default=2048)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--model_type", type=str, default="l3_8b")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=512)


    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.model_type == "l3_8b":
        model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
        gumiho_dir = "amd/Gumiho-llama3-8b"
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    else:
        raise ValueError
    dataset_path = f"gumiho/data/{args.dataset}/question.jsonl"
    max_model_len = args.max_model_len
    prompts = load_prompts(dataset_path, args.num_prompts)
    prompt_ids = [
        tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant."}, 
            {"role": "user", "content": prompt}
        ], add_generation_prompt=True)
        for prompt in prompts
        ]
    llm = LLM(
        model=model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config={
            "method": "gumiho",
            "model": gumiho_dir,
            "num_speculative_tokens": args.num_spec_tokens,
            "draft_tensor_parallel_size": args.draft_tp,
            "max_model_len": max_model_len,
        },
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(temperature=args.temp, max_tokens=args.max_tokens)

    outputs = llm.generate(prompt_token_ids=prompt_ids,
                           sampling_params=sampling_params)

    print("Warmup Done")
    
    outputs = llm.generate(prompt_token_ids=prompt_ids,
                           sampling_params=sampling_params)

    acceptance_counts = [0] * (args.num_spec_tokens + 1)
    for output in outputs:
        for step, count in enumerate(
                output.metrics.spec_token_acceptance_counts):
            acceptance_counts[step] += count

    print("-" * 50)
    print(f"mean acceptance length: \
        {sum(acceptance_counts) / acceptance_counts[0]:.2f}")
    print("-" * 50)


if __name__ == "__main__":
    main()