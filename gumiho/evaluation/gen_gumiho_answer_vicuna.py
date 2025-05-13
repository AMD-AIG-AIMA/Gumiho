# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

from ..model.gumiho_model import GumihoModel
from ..model.kv_cache import initialize_past_key_values
from ..model.utils import *

from loguru import logger
logger.remove()



def run_eval(
        base_model_path,
        gumiho_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        args
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    # random.shuffle(questions)
    shuffled_ids = [q["question_id"] for q in questions]
    # with open(f"data/{args.bench_name}/model_ids/{args.model_id}.shuffled_ids", "w") as fout:
    #     json.dump(shuffled_ids, fout)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                gumiho_model_path,
                model_id,
                questions[i: i + chunk_size],
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                args
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        gumiho_model_path,
        model_id,
        questions,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        args
):
    # temperature = 0.0

    model = GumihoModel.from_pretrained(
        base_model_path=base_model_path,
        gumiho_model_path=gumiho_model_path,
        total_token=args.total_tokens,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # load_in_8bit=True,
        device_map="auto",
        args=args
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        conv = get_conversation_template("vicuna")
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx, _ = model.gumihoGenerate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            conv.stop_str = "</s>"
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    logger.info('Warmup done')

    # questions=questions[6:]
    accept_length_list = []
    speed_list = []
    draft_time_list = []
    for q_i, question in enumerate(tqdm(questions)):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids


                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx, accept_length = model.gumihoGenerate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True
                )
                torch.cuda.synchronize()
                total_time = time.time() - start_time

                # idx = max(idx, 1)
                # accept_length_list.append(new_token/idx)
                # logger.info(f"Q[{q_i}-{j}] accept_length (Current/Mean): {new_token/idx:.3f}/{sum(accept_length_list)/len(accept_length_list):.3f}")
                # # 20250126
                # accept_length_sum = sum(accept_length)
                # logger.info(f"{accept_length_sum=},  {new_token=}")
                # logger.info(f"{len(accept_length)=}, {idx=}")
                # speed_list.append(new_token/total_time)
                # draft_time_list.append(total_time/idx)
                # logger.info(f"speed: {sum(speed_list)/len(speed_list):.3f}")
                # logger.info(f"total_token:{new_token}, total_time:{total_time}")
                # logger.info(f"draft_time:{sum(draft_time_list)/len(draft_time_list)*1000:.3f}ms")
                # logger.info(f" ")

                idx = max(1, idx)
                accept_length_list.append(new_token/(idx))
                logger.info(f"accept_length (Current/Mean): {new_token/(idx):.3f}/{sum(accept_length_list)/len(accept_length_list):.3f}")
                # total_token = sum(accept_length)
                # logger.debug(f"{type(new_token)= }, {type(total_time)= }, {type(new_token/total_time)= }, {(new_token/total_time).dtype= }")
                speed_list.append(new_token/total_time)
                draft_time_list.append(total_time/(idx))
                logger.info(f"speed: {sum(speed_list)/len(speed_list):.3f}")
                logger.info(f"total_token:{new_token}, total_time:{total_time}")
                logger.info(f"draft_time:{sum(draft_time_list)/len(draft_time_list)*1000:.3f}ms")
                logger.warning(f" ")

                output_ids = output_ids[0][len(input_ids[0]):]

                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()


                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gumiho-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/70B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="default")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=100,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--mlp_num', type=int, default=5)
    parser.add_argument('--serial_head_num', type=int, default=2)
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument('--train_mlp_input', type=str, default='ground_truth')  # "ground_truth" "decoder_output"
    parser.add_argument('--mlp_loss_weight', type=float, default=1.0)
    parser.add_argument('--only_accept_max_each_epoch', type=int, default=0)
    parser.add_argument('--run_mode', type=str, default='eval')  # "train" "debug" "eval"
    parser.add_argument('--logger_file', type=str, default='default')
    parser.add_argument('--resume_from', type=int, default=0)
    parser.add_argument('--mlptopk', type=int, default=20)
    parser.add_argument('--pruning', type=int, default=5)
    parser.add_argument('--complete_mask', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='v7b')
    

    args = parser.parse_args()

    log_path = os.path.join("log", f"{args.model_name}")
    os.makedirs(log_path, exist_ok=True)
    logger.add(f"{log_path}/{args.logger_file}.log", level="DEBUG", mode="w", format="{message}")
    print(f"---------> Output to {log_path}/{args.logger_file}.log")

    args_dict = vars(args)
    logger.info("="*30)
    for key, value in args_dict.items():
        logger.info(f"{key}: {value}")
    logger.info("="*30)

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    

    run_eval(
        args.base_model_path,
        args.gumiho_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args
    )

