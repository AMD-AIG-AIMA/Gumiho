# Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding (ICML 2025)
Jinze Li, Yixing Xu, Haiduo Huang, Xuanwu Yin, Dong Li, Edith C.H. Ngai, Emad Barsoum
[Paper](https://arxiv.org/pdf/2503.10135?)


## Introduction

This project implements Gumiho, a novel hybrid architecture designed to accelerate the auto-regressive token generation process of Large Language Models (LLMs) using speculative decoding. Unlike existing methods that treat all tokens within a generated sequence as equally important, Gumiho is based on the theoretical finding that initial tokens in the draft sequence have a more significant impact on the overall accepted length.

Gumiho employs a hybrid head design that combines serial and parallel components. For the crucial early tokens, a sophisticated Transformer architecture is used in a serial configuration to enhance accuracy. For later tokens, multiple lightweight MLP heads operate in parallel to improve efficiency. This approach allocates more computational resources and longer running times to the early heads while utilizing efficient parallel computation for the later ones, leading to improved overall performance and higher acceptance lengths with less processing time per draft-verification cycle. Additionally, Gumiho introduces a Full Tree Attention (FTA) mechanism that enhances the existing Tree Attention by supplementing shorter candidate paths with tokens from longer paths, further increasing the mean accepted tokens without additional computational overhead.


## Performance


**Temperature=0**

| Model  | Method           | MT-Bench Speedup    | MT-Bench $\tau$     | HumanEval Speedup   | HumanEval $\tau$    | GSM8K Speedup    | GSM8K $\tau$    | Alpaca Speedup    | Alpaca $\tau$    | CNN/DM Speedup    | CNN/DM $\tau$    | Natural Ques. Speedup | Natural Ques. $\tau$ | Mean Speedup    | Mean $\tau$    |
| :----: | :---------------: | :-----------------: | :---------------: | :------------------: | :----------------: | :--------------: | :------------: | :---------------: | :------------: | :----------------: | :------------: | :---------------------: | :------------------: | :--------------: | :-----------: |
| V 7B   | Medusa           | 1.96x        | 2.50              | 2.15x         | 2.69               | 2.01x     | 2.59           | 1.94x      | 2.48           | 1.60x       | 2.02           | 1.68x            | 2.05                 | 1.89x     | 2.39          |
| V 7B   | Hydra            | 2.47x        | 3.59              | 2.65x         | 3.78               | 2.49x     | 3.67           | 2.44x      | 3.58           | 1.92x       | 2.70           | 2.01x            | 2.86                 | 2.33x     | 3.36          |
| V 7B   | Eagle            | 2.61x        | 3.82              | 2.96x         | 4.20               | 2.67x     | 4.00           | 2.41x      | 3.66           | 2.35x       | 3.34           | 2.10x            | 3.13                 | 2.52x     | 3.69          |
| V 7B   | Eagle-2          | 2.88x        | 5.00              | 3.27x         | 5.35               | 2.93x     | 4.94           | 2.71x      | 4.85           | 2.45x       | 4.11           | 2.24x            | 3.84                 | 2.74x     | 4.68          |
| V 7B   | **Gumiho(ours)** | **3.15x** | **5.29** | **3.65x** | **5.77** | **3.10x** | **5.06** | **2.83x** | **4.87** | **2.73x** | **4.48** | **2.34x** | **3.88** | **2.97x** | **4.89** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| V 13B  | Medusa           | 2.03x        | 2.58              | 2.24x         | 2.77               | 2.08x     | 2.64           | 2.04x      | 2.44           | 1.67x       | 2.10           | 1.70x            | 2.10                 | 1.96x     | 2.44          |
| V 13B  | Hydra            | 2.65x        | 3.65              | 2.88x         | 3.86               | 2.69x     | 3.67           | 2.65x      | 3.49           | 2.08x       | 2.82           | 2.16x            | 2.86                 | 2.52x     | 3.39          |
| V 13B  | Eagle            | 2.87x        | 3.90              | 3.25x         | 4.29               | 2.88x     | 3.90           | 2.64x      | 3.50           | 2.58x       | 3.49           | 2.21x            | 2.92                 | 2.74x     | 3.66          |
| V 13B  | Eagle-2          | 3.16x        | 4.93              | 3.68x         | 5.42               | 3.19x     | 4.82           | 3.01x      | **4.89** | 2.79x       | 4.27           | 2.41x            | 3.69                 | 3.04x     | 4.67          |
| V 13B  | **Gumiho(ours)** | **3.36x** | **5.16** | **4.11x** | **5.97** | **3.39x** | **5.04** | **3.07x** | 4.88           | **2.91x** | **4.41** | **2.52x** | **3.76** | **3.23x** | **4.87** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 7B  | Eagle            | 2.22x        | 3.00              | 2.53x         | 3.58               | 2.21x     | 3.09           | 2.04x      | 2.88           | 2.08x       | 2.78           | 1.88x            | 2.64                 | 2.16x     | 3.00          |
| L2 7B  | Eagle-2          | 2.91x        | 4.76              | 3.30x         | 5.38               | 2.87x     | 4.76           | 2.81x      | 4.65           | 2.53x       | 4.10           | 2.52x            | 4.16                 | 2.82x     | 4.64          |
| L2 7B  | **Gumiho(ours)** | **3.07x** | **4.90** | **3.55x** | **5.60** | **3.00x** | **4.81** | **2.85x** | **4.55** | **2.66x** | **4.18** | **2.59x** | **4.16** | **2.95x** | **4.70** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 13B | Eagle            | 2.59x        | 3.30              | 2.96x         | 3.90               | 2.61x     | 3.45           | 2.41x      | 3.16           | 2.39x       | 3.09           | 2.15x            | 2.82                 | 2.52x     | 3.29          |
| L2 13B | Eagle-2          | 3.17x        | 4.76              | 3.78x         | 5.53               | 3.23x     | 4.88           | 3.03x      | 4.62           | 2.84x       | 4.27           | 2.76x            | 4.12                 | 3.13x     | 4.70          |
| L2 13B | **Gumiho(ours)** | **3.34x** | **4.98** | **4.05x** | **5.87** | **3.35x** | **5.02** | **3.12x** | **4.66** | **2.93x** | **4.40** | **2.84x** | **4.20** | **3.27x** | **4.85** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 70B | Eagle-2          | 2.51x        | 4.52              | 2.98x         | 5.24               | 2.63x     | 4.63           | 2.48x      | 4.42           | 2.04x       | 3.72           | 2.14x            | 3.88                 | 2.47x     | 4.40          |
| L2 70B | **Gumiho(ours)** | **2.83x** | **4.71** | **3.35x** | **5.43** | **2.90x** | **4.69** | **2.70x** | **4.46** | **2.37x** | **4.08** | **2.35x** | **3.90** | **2.76x** | **4.54** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L3 8B  | Eagle-2          | 2.16x        | 4.36              | 2.51x         | 5.06               | 2.22x     | 4.45           | 2.25x      | 4.88           | 1.82x       | 3.81           | 1.75x            | 3.54                 | 2.12x     | 4.35          |
| L3 8B  | **Gumiho(ours)** | **2.38x** | **4.48** | **2.77x** | **5.18** | **2.49x** | **4.63** | **2.44x** | 4.88           | **2.00x** | **3.94** | **1.93x** | **3.64** | **2.34x** | **4.46** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L3 70B | Eagle-2          | 2.94x        | 4.17              | 3.65x         | 5.09               | 3.17x     | 4.34           | 3.12x      | **4.74** | 2.54x       | 3.66           | 2.48x            | 3.50                 | 2.98x     | 4.25          |
| L3 70B | **Gumiho(ours)** | **3.38x** | **4.28** | **4.28x** | **5.25** | **3.79x** | **4.58** | **3.48x** | 4.58           | **2.91x** | **3.80** | **2.87x** | **3.59** | **3.45x** | **4.35** |

**Temperature=1**

| Model  | Method           | MT-Bench Speedup    | MT-Bench $\tau$     | HumanEval Speedup   | HumanEval $\tau$    | GSM8K Speedup    | GSM8K $\tau$    | Alpaca Speedup    | Alpaca $\tau$    | CNN/DM Speedup    | CNN/DM $\tau$    | Natural Ques. Speedup | Natural Ques. $\tau$ | Mean Speedup    | Mean $\tau$    |
| :----: | :---------------: | :-----------------: | :---------------: | :------------------: | :----------------: | :--------------: | :------------: | :---------------: | :------------: | :----------------: | :------------: | :---------------------: | :------------------: | :--------------: | :-----------: |
| V 7B   | Eagle-2          | 2.51x        | 4.30              | 2.67x         | 4.52               | 2.46x     | 4.47           | 2.38x      | 4.37           | 2.15x       | 3.70           | 2.02x            | 3.50                 | 2.37x     | 4.16          |
| V 7B   | **Gumiho(ours)** | **2.61x** | **4.42** | **2.84x** | **4.62** | **2.73x** | **4.52** | **2.46x** | **4.40** | **2.38x** | **3.94** | **2.10x** | **3.51** | **2.52x** | **4.23** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| V 13B  | Eagle-2          | 2.81x        | 4.37              | 3.32x         | 4.96               | 2.80x     | 4.43           | 2.66x      | 4.46           | 2.51x       | 3.92           | 2.25x            | 3.50                 | 2.73x     | 4.27          |
| V 13B  | **Gumiho(ours)** | **2.93x** | **4.54** | **3.55x** | **5.30** | **2.84x** | **4.59** | **2.77x** | **4.54** | **2.58x** | **4.04** | **2.36x** | **3.72** | **2.84x** | **4.46** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 7B  | Eagle-2          | 2.66x        | 4.63              | 2.95x         | 5.15               | 2.70x     | **4.76** | 2.52x      | 4.40           | 2.34x       | 3.98           | 2.29x            | 4.02                 | 2.58x     | 4.49          |
| L2 7B  | **Gumiho(ours)** | **2.79x** | **4.64** | **3.19x** | **5.27** | **2.78x** | 4.67           | **2.64x** | 4.40           | **2.47x** | **4.05** | **2.44x** | **4.08** | **2.72x** | **4.52** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 13B | Eagle-2          | 3.01x        | 4.60              | 3.58x         | 5.34               | 3.09x     | 4.76           | 2.91x      | 4.49           | 2.71x       | 4.15           | 2.66x            | 4.08                 | 2.99x     | 4.57          |
| L2 13B | **Gumiho(ours)** | **3.18x** | **4.82** | **3.86x** | **5.71** | **3.24x** | **4.94** | **2.98x** | **4.62** | **2.80x** | **4.28** | **2.76x** | **4.16** | **3.14x** | **4.75** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 70B | Eagle-2          | 2.28x        | 4.41              | 2.73x         | 5.15               | 2.42x     | 4.59           | 2.31x      | 4.30           | 1.87x       | 3.67           | 2.00x            | 3.72                 | 2.27x     | 4.30          |
| L2 70B | **Gumiho(ours)** | **2.60x** | **4.65** | **3.15x** | **5.46** | **2.66x** | **4.61** | **2.50x** | **4.43** | **2.15x** | **3.98** | **2.22x** | **3.95** | **2.55x** | **4.51** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L3 8B  | Eagle-2          | 1.93x        | 4.04              | 2.32x         | 4.80               | 2.06x     | 4.27           | 2.03x      | **4.57** | 1.67x       | 3.55           | 1.59x            | 3.27                 | 1.93x     | 4.08          |
| L3 8B  | **Gumiho(ours)** | **2.13x** | **4.14** | **2.55x** | **4.95** | **2.29x** | **4.42** | **2.19x** | 4.55           | **1.86x** | **3.64** | **1.72x** | **3.32** | **2.12x** | **4.17** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L3 70B | Eagle-2          | 2.85x        | 4.07              | 3.57x         | 4.97               | 3.13x     | 4.31           | 3.00x      | **4.65** | 2.47x       | 3.58           | 2.42x            | 3.45                 | 2.91x     | 4.17          |
| L3 70B | **Gumiho(ours)** | **3.29x** | **4.20** | **4.20x** | **5.17** | **3.69x** | **4.49** | **3.34x** | 4.43           | **2.84x** | **3.71** | **2.85x** | **3.57** | **3.37x** | **4.26** |

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
