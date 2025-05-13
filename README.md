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
| V 7B   | Medusa           | 1.96$\times$        | 2.50              | 2.15$\times$         | 2.69               | 2.01$\times$     | 2.59           | 1.94$\times$      | 2.48           | 1.60$\times$       | 2.02           | 1.68$\times$            | 2.05                 | 1.89$\times$     | 2.39          |
| V 7B   | Hydra            | 2.47$\times$        | 3.59              | 2.65$\times$         | 3.78               | 2.49$\times$     | 3.67           | 2.44$\times$      | 3.58           | 1.92$\times$       | 2.70           | 2.01$\times$            | 2.86                 | 2.33$\times$     | 3.36          |
| V 7B   | Eagle            | 2.61$\times$        | 3.82              | 2.96$\times$         | 4.20               | 2.67$\times$     | 4.00           | 2.41$\times$      | 3.66           | 2.35$\times$       | 3.34           | 2.10$\times$            | 3.13                 | 2.52$\times$     | 3.69          |
| V 7B   | Eagle-2          | 2.88$\times$        | 5.00              | 3.27$\times$         | 5.35               | 2.93$\times$     | 4.94           | 2.71$\times$      | 4.85           | 2.45$\times$       | 4.11           | 2.24$\times$            | 3.84                 | 2.74$\times$     | 4.68          |
| V 7B   | **Gumiho(ours)** | **3.15$\times$** | **5.29** | **3.65$\times$** | **5.77** | **3.10$\times$** | **5.06** | **2.83$\times$** | **4.87** | **2.73$\times$** | **4.48** | **2.34$\times$** | **3.88** | **2.97$\times$** | **4.89** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| V 13B  | Medusa           | 2.03$\times$        | 2.58              | 2.24$\times$         | 2.77               | 2.08$\times$     | 2.64           | 2.04$\times$      | 2.44           | 1.67$\times$       | 2.10           | 1.70$\times$            | 2.10                 | 1.96$\times$     | 2.44          |
| V 13B  | Hydra            | 2.65$\times$        | 3.65              | 2.88$\times$         | 3.86               | 2.69$\times$     | 3.67           | 2.65$\times$      | 3.49           | 2.08$\times$       | 2.82           | 2.16$\times$            | 2.86                 | 2.52$\times$     | 3.39          |
| V 13B  | Eagle            | 2.87$\times$        | 3.90              | 3.25$\times$         | 4.29               | 2.88$\times$     | 3.90           | 2.64$\times$      | 3.50           | 2.58$\times$       | 3.49           | 2.21$\times$            | 2.92                 | 2.74$\times$     | 3.66          |
| V 13B  | Eagle-2          | 3.16$\times$        | 4.93              | 3.68$\times$         | 5.42               | 3.19$\times$     | 4.82           | 3.01$\times$      | **4.89** | 2.79$\times$       | 4.27           | 2.41$\times$            | 3.69                 | 3.04$\times$     | 4.67          |
| V 13B  | **Gumiho(ours)** | **3.36$\times$** | **5.16** | **4.11$\times$** | **5.97** | **3.39$\times$** | **5.04** | **3.07$\times$** | 4.88           | **2.91$\times$** | **4.41** | **2.52$\times$** | **3.76** | **3.23$\times$** | **4.87** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 7B  | Eagle            | 2.22$\times$        | 3.00              | 2.53$\times$         | 3.58               | 2.21$\times$     | 3.09           | 2.04$\times$      | 2.88           | 2.08$\times$       | 2.78           | 1.88$\times$            | 2.64                 | 2.16$\times$     | 3.00          |
| L2 7B  | Eagle-2          | 2.91$\times$        | 4.76              | 3.30$\times$         | 5.38               | 2.87$\times$     | 4.76           | 2.81$\times$      | 4.65           | 2.53$\times$       | 4.10           | 2.52$\times$            | 4.16                 | 2.82$\times$     | 4.64          |
| L2 7B  | **Gumiho(ours)** | **3.07$\times$** | **4.90** | **3.55$\times$** | **5.60** | **3.00$\times$** | **4.81** | **2.85$\times$** | **4.55** | **2.66$\times$** | **4.18** | **2.59$\times$** | **4.16** | **2.95$\times$** | **4.70** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 13B | Eagle            | 2.59$\times$        | 3.30              | 2.96$\times$         | 3.90               | 2.61$\times$     | 3.45           | 2.41$\times$      | 3.16           | 2.39$\times$       | 3.09           | 2.15$\times$            | 2.82                 | 2.52$\times$     | 3.29          |
| L2 13B | Eagle-2          | 3.17$\times$        | 4.76              | 3.78$\times$         | 5.53               | 3.23$\times$     | 4.88           | 3.03$\times$      | 4.62           | 2.84$\times$       | 4.27           | 2.76$\times$            | 4.12                 | 3.13$\times$     | 4.70          |
| L2 13B | **Gumiho(ours)** | **3.34$\times$** | **4.98** | **4.05$\times$** | **5.87** | **3.35$\times$** | **5.02** | **3.12$\times$** | **4.66** | **2.93$\times$** | **4.40** | **2.84$\times$** | **4.20** | **3.27$\times$** | **4.85** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 70B | Eagle-2          | 2.51$\times$        | 4.52              | 2.98$\times$         | 5.24               | 2.63$\times$     | 4.63           | 2.48$\times$      | 4.42           | 2.04$\times$       | 3.72           | 2.14$\times$            | 3.88                 | 2.47$\times$     | 4.40          |
| L2 70B | **Gumiho(ours)** | **2.83$\times$** | **4.71** | **3.35$\times$** | **5.43** | **2.90$\times$** | **4.69** | **2.70$\times$** | **4.46** | **2.37$\times$** | **4.08** | **2.35$\times$** | **3.90** | **2.76$\times$** | **4.54** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L3 8B  | Eagle-2          | 2.16$\times$        | 4.36              | 2.51$\times$         | 5.06               | 2.22$\times$     | 4.45           | 2.25$\times$      | 4.88           | 1.82$\times$       | 3.81           | 1.75$\times$            | 3.54                 | 2.12$\times$     | 4.35          |
| L3 8B  | **Gumiho(ours)** | **2.38$\times$** | **4.48** | **2.77$\times$** | **5.18** | **2.49$\times$** | **4.63** | **2.44$\times$** | 4.88           | **2.00$\times$** | **3.94** | **1.93$\times$** | **3.64** | **2.34$\times$** | **4.46** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L3 70B | Eagle-2          | 2.94$\times$        | 4.17              | 3.65$\times$         | 5.09               | 3.17$\times$     | 4.34           | 3.12$\times$      | **4.74** | 2.54$\times$       | 3.66           | 2.48$\times$            | 3.50                 | 2.98$\times$     | 4.25          |
| L3 70B | **Gumiho(ours)** | **3.38$\times$** | **4.28** | **4.28$\times$** | **5.25** | **3.79$\times$** | **4.58** | **3.48$\times$** | 4.58           | **2.91$\times$** | **3.80** | **2.87$\times$** | **3.59** | **3.45$\times$** | **4.35** |

**Temperature=1**

| Model  | Method           | MT-Bench Speedup    | MT-Bench $\tau$     | HumanEval Speedup   | HumanEval $\tau$    | GSM8K Speedup    | GSM8K $\tau$    | Alpaca Speedup    | Alpaca $\tau$    | CNN/DM Speedup    | CNN/DM $\tau$    | Natural Ques. Speedup | Natural Ques. $\tau$ | Mean Speedup    | Mean $\tau$    |
| :----: | :---------------: | :-----------------: | :---------------: | :------------------: | :----------------: | :--------------: | :------------: | :---------------: | :------------: | :----------------: | :------------: | :---------------------: | :------------------: | :--------------: | :-----------: |
| V 7B   | Eagle-2          | 2.51$\times$        | 4.30              | 2.67$\times$         | 4.52               | 2.46$\times$     | 4.47           | 2.38$\times$      | 4.37           | 2.15$\times$       | 3.70           | 2.02$\times$            | 3.50                 | 2.37$\times$     | 4.16          |
| V 7B   | **Gumiho(ours)** | **2.61$\times$** | **4.42** | **2.84$\times$** | **4.62** | **2.73$\times$** | **4.52** | **2.46$\times$** | **4.40** | **2.38$\times$** | **3.94** | **2.10$\times$** | **3.51** | **2.52$\times$** | **4.23** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| V 13B  | Eagle-2          | 2.81$\times$        | 4.37              | 3.32$\times$         | 4.96               | 2.80$\times$     | 4.43           | 2.66$\times$      | 4.46           | 2.51$\times$       | 3.92           | 2.25$\times$            | 3.50                 | 2.73$\times$     | 4.27          |
| V 13B  | **Gumiho(ours)** | **2.93$\times$** | **4.54** | **3.55$\times$** | **5.30** | **2.84$\times$** | **4.59** | **2.77$\times$** | **4.54** | **2.58$\times$** | **4.04** | **2.36$\times$** | **3.72** | **2.84$\times$** | **4.46** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 7B  | Eagle-2          | 2.66$\times$        | 4.63              | 2.95$\times$         | 5.15               | 2.70$\times$     | **4.76** | 2.52$\times$      | 4.40           | 2.34$\times$       | 3.98           | 2.29$\times$            | 4.02                 | 2.58$\times$     | 4.49          |
| L2 7B  | **Gumiho(ours)** | **2.79$\times$** | **4.64** | **3.19$\times$** | **5.27** | **2.78$\times$** | 4.67           | **2.64$\times$** | 4.40           | **2.47$\times$** | **4.05** | **2.44$\times$** | **4.08** | **2.72$\times$** | **4.52** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 13B | Eagle-2          | 3.01$\times$        | 4.60              | 3.58$\times$         | 5.34               | 3.09$\times$     | 4.76           | 2.91$\times$      | 4.49           | 2.71$\times$       | 4.15           | 2.66$\times$            | 4.08                 | 2.99$\times$     | 4.57          |
| L2 13B | **Gumiho(ours)** | **3.18$\times$** | **4.82** | **3.86$\times$** | **5.71** | **3.24$\times$** | **4.94** | **2.98$\times$** | **4.62** | **2.80$\times$** | **4.28** | **2.76$\times$** | **4.16** | **3.14$\times$** | **4.75** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L2 70B | Eagle-2          | 2.28$\times$        | 4.41              | 2.73$\times$         | 5.15               | 2.42$\times$     | 4.59           | 2.31$\times$      | 4.30           | 1.87$\times$       | 3.67           | 2.00$\times$            | 3.72                 | 2.27$\times$     | 4.30          |
| L2 70B | **Gumiho(ours)** | **2.60$\times$** | **4.65** | **3.15$\times$** | **5.46** | **2.66$\times$** | **4.61** | **2.50$\times$** | **4.43** | **2.15$\times$** | **3.98** | **2.22$\times$** | **3.95** | **2.55$\times$** | **4.51** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L3 8B  | Eagle-2          | 1.93$\times$        | 4.04              | 2.32$\times$         | 4.80               | 2.06$\times$     | 4.27           | 2.03$\times$      | **4.57** | 1.67$\times$       | 3.55           | 1.59$\times$            | 3.27                 | 1.93$\times$     | 4.08          |
| L3 8B  | **Gumiho(ours)** | **2.13$\times$** | **4.14** | **2.55$\times$** | **4.95** | **2.29$\times$** | **4.42** | **2.19$\times$** | 4.55           | **1.86$\times$** | **3.64** | **1.72$\times$** | **3.32** | **2.12$\times$** | **4.17** |
|        |                  |                     |                   |                      |                    |                  |                |                   |                |                    |                |                         |                      |                 |               |
| L3 70B | Eagle-2          | 2.85$\times$        | 4.07              | 3.57$\times$         | 4.97               | 3.13$\times$     | 4.31           | 3.00$\times$      | **4.65** | 2.47$\times$       | 3.58           | 2.42$\times$            | 3.45                 | 2.91$\times$     | 4.17          |
| L3 70B | **Gumiho(ours)** | **3.29$\times$** | **4.20** | **4.20$\times$** | **5.17** | **3.69$\times$** | **4.49** | **3.34$\times$** | 4.43           | **2.84$\times$** | **3.71** | **2.85$\times$** | **3.57** | **3.37$\times$** | **4.26** |



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
