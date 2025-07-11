
max_num_seqs=1
max_model_len=2048
max_tokens=512
model_type="l3_8b"
ds='gsm8k'
spd_tokens=3

export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'

 
python gumiho/vllm_gumiho/vllm.py \
    --max_tokens ${max_tokens} \
    --max_num_seqs ${max_num_seqs} \
    --max_model_len ${max_model_len} \
    --model_type ${model_type} \
    --num_spec_tokens ${spd_tokens} \
    --dataset ${ds}



