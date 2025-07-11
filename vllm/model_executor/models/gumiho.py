# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

from typing import Iterable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import maybe_prefix

logger = init_logger(__name__)


class DummyInputLayerNorm(nn.Module):

    def __init__(self, weight=None, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight) if weight is not None else None
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        return x


class DummyOutputNorm(nn.Module):

    def forward(self, x, residual):
        if residual is None:
            return x
        else:
            return x + residual, None


class ResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        # Handling residual connection when reducing dim
        if input_size != output_size:
            self.res_connection = nn.Linear(input_size, output_size)
        else:
            self.res_connection = nn.Identity()
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.res_connection(x) + self.act(self.linear(x))

class noResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.linear(x))



class Gumiho(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config

        architectures = getattr(self.config.model, "architectures", [])
        model_cls, _ = ModelRegistry.resolve_model_cls(architectures)

        self.model = model_cls(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))

        
        self.fc = nn.Linear(config.model.hidden_size * 2,
                            config.model.hidden_size,
                            bias=getattr(self.config, "gumiho_fc_bias", False))

        # Modify layer normalization and residual connections as suggested
        # in the EAGLE framework: https://github.com/SafeAILab/EAGLE
        # While weights and biases are generally not needed,
        # they are retained here to support certain unit tests
        # (e.g., spec_decode/e2e/test_eagle_correctness.py).
        if not hasattr(self.config.model,
                       "skip_prenorm") or self.config.model.skip_prenorm:
            self.model.model.layers[0].input_layernorm = DummyInputLayerNorm(
                weight=self.model.model.layers[0].input_layernorm.weight)

        if not hasattr(
                self.config.model,
                "skip_output_norm") or self.config.model.skip_output_norm:
            self.model.model.norm = DummyOutputNorm()

        self.add_para_norm = False
        if hasattr(self.config.model,
                   "add_para_norm") and self.config.model.add_para_norm:
            self.enorm = RMSNorm(config.model.hidden_size, eps=config.rms_norm_eps)
            self.hnorm = RMSNorm(config.model.hidden_size, eps=config.rms_norm_eps)
            self.add_para_norm = True

        self.orig_vocab_size = config.vocab_size
        self.truncated_vocab_size = config.truncated_vocab_size
        self.unpadded_vocab_size = self.truncated_vocab_size

        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.model.hidden_size,
            org_num_embeddings=self.truncated_vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                self.truncated_vocab_size,
                                                logit_scale)

        
        self.token_map = None

        self.mlp = nn.ModuleList([
                        nn.Sequential(
                            noResBlock(config.model.hidden_size * 2, config.model.hidden_size * 1),
                            ResBlock(config.model.hidden_size, config.model.hidden_size),
                            ResBlock(config.model.hidden_size, config.model.hidden_size),
                            ResBlock(config.model.hidden_size, config.model.hidden_size),
                            ResBlock(config.model.hidden_size, config.model.hidden_size),
                            ResBlock(config.model.hidden_size, config.model.hidden_size)
                        ) for _ in range(vllm_config.speculative_config.num_speculative_tokens-2)
                    ])

    def mlp_forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        return [block(hidden_states) for block in self.mlp]

    def mlp_compute_logits(
            self, hidden_states: List[torch.Tensor],
            sampling_metadata: SamplingMetadata) -> List[torch.Tensor]:
        logits_lst: List[torch.Tensor] = []

        lm_head = self.lm_head
        for hs in hidden_states:
            _logits = self.logits_processor(lm_head, hs, sampling_metadata)

            if _logits is None:
                # _logits should only be None on rank > 0, in which case
                # it should remain true for every lm_head
                assert len(logits_lst) == 0
                continue

            if self.token_map is None:
                logits_lst.append(_logits)
            else:
                logits_lst.append(-torch.inf * torch.ones(
                    size=(*_logits.shape[:-1], self.orig_vocab_size),
                    device=_logits.device,
                    dtype=_logits.dtype))

                logits_lst[-1][..., self.token_map] = _logits

        return logits_lst

    def mlp_sample(
        self,
        logits: List[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> List[SamplerOutput]:
        logits = torch.stack(logits, dim=0).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        token_ids = logits.argmax(-1)  # support only top-1 for now
        probs = torch.softmax(logits, dim=-1)

        token_id_list = []
        token_prob_list = []
        token_logprob_list = []
        outputs: List[Optional[SamplerOutput]] = []

        for idx, seq_group in enumerate(sampling_metadata.seq_groups):
            token_id_list.append(token_ids[:, seq_group.sample_indices])
            token_prob_list.append(probs[:, seq_group.sample_indices])
            token_logprob_list.append(logprobs[:, seq_group.sample_indices])

        for idx in range(token_prob_list[0].shape[0]):
            outputs.append(
                SamplerOutput(
                    outputs=None,
                    sampled_token_probs=probs[idx, :],
                    logprobs=logprobs[idx, :],
                    sampled_token_ids=token_ids[idx, :],
                ))

        # outputs: List[Optional[SamplerOutput]] = []
        # for idx in range(len(sampling_metadata.seq_groups)):
        #     outputs.append(
        #         SamplerOutput(
        #             outputs=None,
        #             sampled_token_probs=token_prob_list[idx].squeeze(1),
        #             logprobs=token_logprob_list[idx].squeeze(1),
        #             sampled_token_ids=token_id_list[idx].squeeze(1),
        #         ))

        return outputs

    def generate_mlp_proposals(
        self,
        previous_hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> List[SamplerOutput]:

        mlp_input = []
        for idx in range(len(previous_hidden_states)):
            _hidden_states = previous_hidden_states[idx].hidden_states
            _ids = previous_hidden_states[idx].sampled_token_ids.squeeze(0)
            _id_emb = self.get_input_embeddings(_ids)
            _id_emb = torch.cat([_id_emb, _hidden_states], dim=-1)
            inputs_embeds = self.fc(_id_emb)
            mlp_input.append(inputs_embeds)

        mlp_input = torch.cat(mlp_input, dim=-1)



        return self.mlp_sample(
            logits=self.mlp_compute_logits(
                hidden_states=self.mlp_forward(mlp_input),
                sampling_metadata=sampling_metadata,
            ),
            sampling_metadata=sampling_metadata,
        )

    @property
    def sampler(self):
        return self.model.sampler

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        if self.add_para_norm:
            inputs_embeds = torch.cat([
                self.enorm(inputs_embeds),
                self.hnorm(previous_hidden_states)
            ],
                                      dim=-1)
        else:
            inputs_embeds = torch.cat([inputs_embeds, previous_hidden_states],
                                      dim=-1)

        inputs_embeds = self.fc(inputs_embeds)

        inputs_embeds[positions == 0] = 0  # masking inputs at position=0

        hidden_states = self.model.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
        )
        return hidden_states
    
    

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        if self.token_map is not None:
            _logits = logits
            logits = -torch.inf * torch.ones(
                size=(*_logits.shape[:-1], self.orig_vocab_size),
                device=_logits.device,
                dtype=_logits.dtype)

            logits[..., self.token_map] = _logits

        return (logits, hidden_states)

    def sample(
        self,
        logits,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        _logits, _hidden_states = logits
        next_tokens = self.sampler(_logits, sampling_metadata)
        next_tokens.hidden_states = _hidden_states[-(_logits.shape[0]):, :]
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # This implementation is incompitable with https://huggingface.co/yuhuili/EAGLE-LLaMA3-Instruct-8B
        # due to missing lm_head weights and its config being that of a
        # Llama model. Here's a compatible version with the same weights:
        # https://huggingface.co/abhigoyal/EAGLE-LLaMA3-Instruct-8B-vllm
        # Also, here's an example script for converting trained EAGLE
        # checkpoint to vLLM compatible version: https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d
        # print(f"{weights=}")
        from transformers import AutoModelForCausalLM
        lm_head_weight_path = "meta-llama/Meta-Llama-3-8B-Instruct"
        model_for_lm_head = AutoModelForCausalLM.from_pretrained(
            lm_head_weight_path,      
            trust_remote_code=True    
        )
        lm_head_weight = {}
        for name, weight in model_for_lm_head.state_dict().items():
            if name.startswith("lm_head.weight"):
                lm_head_weight["weight"] = weight


        model_weights = {}
        weights = list(weights)
        for name, loaded_weight in weights:
            if name == "token_map":
                if self.config.truncated_vocab_size < self.config.vocab_size:
                    self.token_map = nn.Parameter(loaded_weight,
                                                  requires_grad=False)
            elif name.startswith("fc.weight"):
                weight_loader = getattr(self.fc.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.fc.weight, loaded_weight)
            elif name.startswith("fc.bias"):
                if self.fc.bias is not None:
                    weight_loader = getattr(self.fc.bias, "weight_loader",
                                            default_weight_loader)
                    weight_loader(self.fc.bias, loaded_weight)
                else:
                    logger.warning_once("Found bias in the loaded weights but "
                                        "the model config doesn't have bias.")
            elif name.startswith("enorm.weight"):
                weight_loader = getattr(self.enorm.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.enorm.weight, loaded_weight)
            elif name.startswith("hnorm.weight"):
                weight_loader = getattr(self.hnorm.weight, "weight_loader",
                                        default_weight_loader)
                weight_loader(self.hnorm.weight, loaded_weight)
            elif name.startswith("model.lm_head.") or name.startswith(
                    "model.model."):
                model_weights[name.split("model.", 1)[-1]] = loaded_weight
            elif name.startswith("lm_head.") or name.startswith("model."):
                model_weights[name] = loaded_weight
            elif name.startswith("mlp."):
                # print(f"Will not load mlp parameters in the first two headers")
                pass
            else:
                model_weights[f"model.{name}"] = loaded_weight

        if len(lm_head_weight) == 0:
            lm_head_weight = torch.zeros(
                self.lm_head.org_vocab_size,
                self.lm_head.embedding_dim,
                dtype=self.config.torch_dtype,
            )
            raise ValueError

        # weight_loader = getattr(self.lm_head.weight, "weight_loader",
        #                         default_weight_loader)
        # weight_loader(self.lm_head.weight, lm_head_weight)
        missing, unexpected = self.lm_head.load_state_dict(lm_head_weight, strict=False)

        self.model.load_weights(model_weights.items())

        # load mlp weight
        prefix = "mlp."
        mlp_weights = {
                        name[len(prefix):]: tensor
                        for name, tensor in weights
                        if name.startswith(prefix)
                    }
        
        missing, unexpected = self.mlp.load_state_dict(mlp_weights, strict=False)

        if missing:
            print("The following parameters were not loaded (missing in the model):", missing)
        if unexpected:
            print("The following parameters are not part of mlp and are ignored:", unexpected)

