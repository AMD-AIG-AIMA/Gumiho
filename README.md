# Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding (ICML 2025)
Jinze Li, Yixing Xu, Haiduo Huang, Xuanwu Yin, Dong Li, Edith C.H. Ngai, Emad Barsoum
[Paper](https://arxiv.org/pdf/2503.10135?)


## Introduction

This project implements Gumiho, a novel hybrid architecture designed to accelerate the auto-regressive token generation process of Large Language Models (LLMs) using speculative decoding. Unlike existing methods that treat all tokens within a generated sequence as equally important, Gumiho is based on the theoretical finding that initial tokens in the draft sequence have a more significant impact on the overall accepted length.

Gumiho employs a hybrid head design that combines serial and parallel components. For the crucial early tokens, a sophisticated Transformer architecture is used in a serial configuration to enhance accuracy. For later tokens, multiple lightweight MLP heads operate in parallel to improve efficiency. This approach allocates more computational resources and longer running times to the early heads while utilizing efficient parallel computation for the later ones, leading to improved overall performance and higher acceptance lengths with less processing time per draft-verification cycle. Additionally, Gumiho introduces a Full Tree Attention (FTA) mechanism that enhances the existing Tree Attention by supplementing shorter candidate paths with tokens from longer paths, further increasing the mean accepted tokens without additional computational overhead.




## Checkpoints (ckpt)

We provide checkpoints for Gumiho trained on various base LLMs. You can find the download links in the table below:

| Model         | Checkpoint Link             |
|---------------|-----------------------------|
| Vicuna 7B     | [Placeholder for ckpt link] |
| Vicuna 13B    | [Placeholder for ckpt link] |
| Llama2 7B     | [Placeholder for ckpt link] |
| Llama2 13B    | [Placeholder for ckpt link] |
| Llama2 70B    | [Placeholder for ckpt link] |
| Llama3 8B     | [Placeholder for ckpt link] |
| Llama3 70B    | [Placeholder for ckpt link] |

## Environment Setup

Setting up the environment is straightforward. We provide a `requirements.txt` file listing all necessary dependencies. Simply run the following command:

```bash
pip install -r requirements.txt
```


## Evaluation

To evaluate the method, we provide a script folder. The script in it contains the necessary commands and hyperparameters for evaluation. You can run evaluation directly by executing the script:
```bash
bash scripts/eval_llama3.sh
```

## Training

For training the Gumiho model, we provide a train.sh script. This script includes the commands and hyperparameters for the training process. To start training, run:
```bash
bash scripts/train.sh
```

## Citation
```
@article{li2025gumiho,
  title={Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding},
  author={Li, Jinze and Xu, Yixing and Huang, Haiduo and Yin, Xuanwu and Li, Dong and Ngai, Edith CH and Barsoum, Emad},
  journal={arXiv preprint arXiv:2503.10135},
  year={2025}
}
```

## Acknowledgements
This project is built upon [Eagle](https://github.com/SafeAILab/EAGLE), with additional modifications and extensions.
