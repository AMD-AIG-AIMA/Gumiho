cd ./gumiho/train

logger_name="logger_name_to_be_set"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed main_deepspeed.py --deepspeed --deepspeed_config ds_config.json \
    --tmpdir "pre_generated_data" \
    --cpdir "path_to_save_ckpt" \
    --configpath "../gumiho/train/Gumiho-LLaMA3-Instruct-70B.json" \
    --basepath "target_model_path" \
    --logger_file ${logger_name} \
    --p_w 0.1 \
    --mlp_v_w 1.0 \
    --mlp_p_w 100 \
    --max_len 2048 \
    --model_name l3_70b \
    --start_epoch 0 
