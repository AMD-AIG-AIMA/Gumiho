#!/bin/bash
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.

mlptopk=35
tt=64
d=6
mlpnum=5
p=35
topk=14
complete_mask=1
model_name="l3_8b"  # l3_8b l3_70b
temp=0  # 0 1
bench_name=gsm8k  # alpaca gsm8k humaneval mt_bench qa sum

if [[ "$model_name" == "l3_8b" ]]; then
    gumiho_path="l3_8b_ckpt"
    base_model_path="meta-llama/Meta-Llama-3-8B-Instruct"
elif [[ "$model_name" == "l3_70b" ]]; then
    gumiho_path="l3_70b_ckpt"
    base_model_path="meta-llama/Meta-Llama-3-70B-Instruct"
else
    echo "Error: Unknown model_name '$model_name'"
    exit 1
fi

for trial in 0 1
do
    CUDA_VISIBLE_DEVICES=7 python -m gumiho.evaluation.gen_gumiho_answer_llama3chat \
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
