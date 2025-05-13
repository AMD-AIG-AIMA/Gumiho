#!/bin/bash

mlptopk=35
tt=64
d=6
mlpnum=5
p=35
topk=14
complete_mask=1
model_name="v7b"  # v7b v13b
temp=0  # 0 1
bench_name=gsm8k  # alpaca gsm8k humaneval mt_bench qa sum

if [[ "$model_name" == "v7b" ]]; then
    gumiho_path="v7b_ckpt"
    base_model_path="lmsys/vicuna-7b-v1.3"
elif [[ "$model_name" == "v13b" ]]; then
    gumiho_path="v13b_ckpt"
    base_model_path="lmsys/vicuna-13b-v1.3"
else
    echo "Error: Unknown model_name '$model_name'"
    exit 1
fi

for trial in 0 1
do
    CUDA_VISIBLE_DEVICES=7 python -m gumiho.evaluation.gen_gumiho_answer_vicuna \
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
