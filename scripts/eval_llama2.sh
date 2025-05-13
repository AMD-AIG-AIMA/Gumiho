#!/bin/bash
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

mlptopk=35
tt=64
d=6
mlpnum=5
p=35
topk=14
complete_mask=1
model_name="l2_7b"  # l2_13b l2_7b l2_70b
temp=0  # 0 1
bench_name=gsm8k  # alpaca gsm8k humaneval mt_bench qa sum

if [[ "$model_name" == "l2_7b" ]]; then
    gumiho_path="l2_7b_ckpt"
    base_model_path="meta-llama/Llama-2-7b-chat-hf"
elif [[ "$model_name" == "l2_13b" ]]; then
    gumiho_path="l2_13b_ckpt"
    base_model_path="meta-llama/Llama-2-13b-chat-hf"
elif [[ "$model_name" == "l2_70b" ]]; then
    gumiho_path="l2_70b_ckpt"
    base_model_path="meta-llama/Llama-2-70b-chat-hf"
else
    echo "Error: Unknown model_name '$model_name'"
    exit 1
fi

for trial in 0 1
do
    CUDA_VISIBLE_DEVICES=7 python -m gumiho.evaluation.gen_gumiho_answer_llama2chat \
      --gumiho-model-path ${gumiho_path} \
      --base-model-path ${base_model_path} \
      --mlptopk $mlptopk \
      --top-k $topk \
      --total-tokens $tt \
      --depth $d \
      --mlp_num $mlpnum \
      --pruning $p \
      --complete_mask $complete_mask \
      --bench-name $bench_name \
      --temperature $temp \
      --model_name $model_name \
      --logger_file ${model_name}_${bench_name}_temp${temp}_trial${trial}
done
