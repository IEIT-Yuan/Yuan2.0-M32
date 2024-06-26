# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Yuan model compatible with HuggingFace weights."""
import pdb
from typing import Any, Dict, Iterable, List, Optional, Tuple
import copy
import math
import numpy as np
import torch
from torch import einsum, nn
from vllm.model_executor.models.configuration_yuan import YuanConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.config import LoRAConfig
from transformers.activations import ACT2FN
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.utils import set_weight_attrs
from vllm.attention import Attention, AttentionMetadata
from apex.normalization import MixedFusedRMSNorm as RMSNorm
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.distributed import (get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.model_loader.weight_utils import (
            default_weight_loader, kv_cache_scales_loader)
from vllm.sequence import SamplerOutput
from vllm._C import ops
from vllm._C import cache_ops

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

LFCache = Tuple[torch.Tensor, torch.Tensor]


class YuanRouter(nn.Module):
    """A Router implementation for DBRX that returns logits for each expert
    per token.
    """

    def __init__(
        self,
        config: YuanConfig,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.moe_config['moe_num_experts']
        self.hidden_size = config.hidden_size
        self.layer = ColumnParallelLinear(
            self.hidden_size,
            self.num_total_experts,
            bias=False,
            params_dtype=params_dtype,
            linear_method=None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.layer(hidden_states)
        return router_logits

class ParallelAttention_router(nn.Module):
    def __init__(self, config):
        super(ParallelAttention_router, self).__init__()

        self.hidden_size = config.hidden_size
        self.projection_size = config.moe_config['moe_num_experts']
        self.query_key_value = ReplicatedLinear(self.hidden_size, self.projection_size*3, bias=False, linear_method=None)

    def forward(self, hidden_states, attn_metadata):
        data_type = hidden_states.dtype
        mix_layer, _ = self.query_key_value(hidden_states)
        (query_layer, key_layer, value_layer) = torch.chunk(mix_layer, 3, dim=-1)
        
        query_layer = query_layer.view(*query_layer.shape, 1).float()
        key_layer = key_layer.view(*key_layer.shape, 1).float()
        value_layer = value_layer.view(*value_layer.shape, 1).float()

        attn_weights = torch.matmul(query_layer, key_layer.transpose(1,2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value_layer)
        router_output = attn_output.squeeze(2)
        return router_output

class YuanExperts(nn.Module):
    """A tensor-parallel MoE implementation for DBRX.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(
        self,
        config: YuanConfig,
        linear_method: Optional[LinearMethodBase] = None,
        params_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_total_experts = config.moe_config['moe_num_experts']
        self.top_k = config.moe_config['moe_top_k']
        self.hidden_size = config.hidden_size 
        self.intermediate_size = (config.moe_config['ffn_hidden_size'] //
                                  self.tp_size)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        self.router = ParallelAttention_router(config)
        self.ws = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                2 * self.intermediate_size,
                self.hidden_size,
                device="cuda",
                dtype=self.params_dtype,
            ))
        self.w2s = nn.Parameter(
            torch.empty(
                self.num_total_experts,
                self.hidden_size,
                self.intermediate_size,
                device="cuda",
                dtype=self.params_dtype,
            ))

        set_weight_attrs(
            self.ws,
            {
                "weight_loader": self.weight_loader,
            },
        )
        set_weight_attrs(
            self.w2s,
            {
                "weight_loader": self.weight_loader,
            },
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        if weight_name.endswith("ws"):
            chunk_size = loaded_weight.shape[2] // 2
            chunk0 = torch.split(loaded_weight, chunk_size, dim=2)[0].clone().detach()
            chunk1 = torch.split(loaded_weight, chunk_size, dim=2)[1].clone().detach()
            sub_chunk_size = param_data.shape[1] // 2
            sub_chunk0 = torch.split(chunk0, sub_chunk_size, dim=2)[tp_rank].clone().detach()
            sub_chunk1 = torch.split(chunk1, sub_chunk_size, dim=2)[tp_rank].clone().detach()
            param_data.copy_(torch.cat([sub_chunk0, sub_chunk1], dim=2).permute(0, 2, 1))
            
        if weight_name.endswith("w2s"):
            chunk_size = loaded_weight.shape[1] // self.tp_size
            sub_chunk = torch.split(loaded_weight, chunk_size, dim=1)[tp_rank].clone().detach()
            param_data.copy_(sub_chunk.permute(0, 2, 1))

    def forward(self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata) -> torch.Tensor:
        router_logits = self.router(hidden_states, attn_metadata)
        final_hidden_states = fused_moe(
            hidden_states,
            self.ws,
            self.w2s,
            router_logits,
            self.top_k,
            renormalize=False,
            inplace=True,
            no_moe_kernels=True,
            topk_before_softmax=True,
            model_type='yuan'
        )

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)
        return final_hidden_states

def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)

def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class YuanYaRNScaledRotaryEmbedding(nn.Module):
    def __init__(self, dim, rotary_base=10000, max_position_embeddings=2048, scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, dtype=torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = rotary_base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        self.revised_yarn()
        self.max_seq_len_cached = max_position_embeddings
        t = np.arange(self.max_seq_len_cached)
        t = torch.tensor(t, device=self.inv_freq.device,dtype=torch.float)
        freqs = torch.outer(t, self.inv_freq)
        self.emb = torch.cat((freqs, freqs), dim=-1)

    def forward(self, x, seq_len=None):
        return self.emb[:, None, None, :]

    def yarn(self, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)
        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float().to(device)) * self.extrapolation_factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def revised_yarn(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).float()) * self.extrapolation_factor
        inv_freq = inv_freq / ((1-inv_freq_mask)*self.scale + inv_freq_mask)
        self.register_buffer("inv_freq", inv_freq)


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, freqs, position_ids, use_yarn, yarn_scale_factor, attn_factor, attn_metadata):
    data_type = x.dtype
    rot_dim = freqs.shape[-1]
    freqs = freqs[position_ids]
    # feqs [b*s, 1, 1, head_dim]
    freqs = freqs.view(x.shape[0],freqs.shape[1],freqs.shape[2])
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    mscale = float(_yarn_get_mscale(yarn_scale_factor) * attn_factor) if use_yarn else 1.0
    x = (x * freqs.cos() * mscale) + (_rotate_half(x) * freqs.sin() * mscale)
    return torch.cat((x, x_pass), dim=-1).to(data_type)

class YuanRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.base = base
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        freqs = einsum('i , j -> i j', seq.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:, None, :].float()

class LocalizedFiltering(torch.nn.Module):
    """
    Mega's Exponential Moving Average layer, largely left unmodified from the original repo with the exception of
    variable names and moving away from the stateful representation of incremental decoding state. See
    "https://arxiv.org/abs/2209.10655" for more details.
    """

    def __init__(self, config, hidden_size):
        super().__init__()
        self.embed_dim = hidden_size
        self.lf_conv2d_group = 1
        self.lf_conv2d_num_pad = 0
        self.conv1 = torch.nn.Conv2d(self.embed_dim, self.embed_dim // 2, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.conv2 = torch.nn.Conv2d(self.embed_dim // 2, self.embed_dim, (2, 1), stride=(1, 1), padding=(self.lf_conv2d_num_pad, 0), groups=self.lf_conv2d_group)
        self.output_layernorm = RMSNorm(self.embed_dim, eps=config.rms_norm_eps)

    def forward(self, inputs, lf1_cache, lf2_cache):
       inputs = inputs.permute([1, 0, 2]) # [ s, b, h]
       residual = inputs
       old_shape = inputs.shape
       new_shape = inputs.view(inputs.shape[0], 1, inputs.shape[1], inputs.shape[2]).shape # [s, 1, b, h]

       inputs = inputs.view(new_shape).permute([2, 3, 0, 1]) # [b, h, s, 1]
       inputs = torch.cat([lf1_cache, inputs], dim=2) # [b, h, s+1, 1]
       output1 = self.conv1(inputs)
       output1 = torch.cat([lf2_cache, output1], dim=2)
       output2 = self.conv2(output1).permute([2, 3, 0, 1])
       output2 = output2.view(old_shape)
       
       assert list(output2.shape) == list(residual.shape), f'{output2.shape}, {residual.shape}'
       output3 = output2 + residual
       lf_output = self.output_layernorm(output3)
       lf_output = lf_output.permute([1, 0, 2])

       lf1_cache = inputs[:, :, -1:, :].contiguous()
       lf2_cache = output1[:, :, -1:, :].contiguous()
       return lf_output, lf1_cache, lf2_cache

class YuanMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.up_proj = ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=False,
                                            linear_method=linear_method)
        self.gate_proj= ColumnParallelLinear(hidden_size,
                                            intermediate_size,
                                            bias=False,
                                            linear_method=linear_method)
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           linear_method=linear_method)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x, attn_metadata: AttentionMetadata):
        x1, _ = self.up_proj(x)
        x3 = self.act_fn(x1)
        x2, _ = self.gate_proj(x)
        x, _ = self.down_proj(x2 * x3)
        return x


class YuanAttention(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        hidden_size: int,
        attention_projection_size: int,
        num_heads: int,
        num_kv_heads=None,
        attn_head_size=None,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        self.attn_head_size = attention_projection_size // num_heads if attn_head_size is None else attn_head_size
        assert self.total_num_heads % self.tp_size == 0
        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.eps = 1e-6
        self.qk_proj = ColumnParallelLinear(
            hidden_size,
            2 * attention_projection_size,
            bias=False,
            linear_method=linear_method,
        )
        self.v_proj = ColumnParallelLinear(
            hidden_size,
            attention_projection_size,
            bias=False,
            linear_method=linear_method,
        )
        self.o_proj = RowParallelLinear(
            attention_projection_size,
            hidden_size,
            bias=False,
            linear_method=linear_method,
        )
        
        self.model_type = getattr(config, 'model_type', 'yuan')
        self.lf_gate = LocalizedFiltering(self.config, self.hidden_size)
        self.attn = Attention(self.num_kv_heads,
                              self.attn_head_size,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        kv_cache: torch.Tensor,
        lf1_cache: torch.Tensor,
        lf2_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        use_yarn: bool=False,
        yarn_scale_factor: float=1.0,
        attn_factor: float=1.0,
    ) -> torch.Tensor:
        q_len, hidden_size = hidden_states.size()
        bsz = attn_metadata.num_prefills + attn_metadata.num_decode_tokens
        
        v, _ = self.v_proj(hidden_states)
        v = v.view(*v.shape[:-1], self.num_heads, self.attn_head_size)
        result = []
        if attn_metadata.prefill_metadata != None:
            for b in range(bsz):
                tmp_hidden_states, lf1, lf2 = self.lf_gate(hidden_states[attn_metadata.prefill_metadata.seq_start_loc[b]:attn_metadata.prefill_metadata.seq_start_loc[b+1]].unsqueeze(0), lf1_cache[b:b+1], lf2_cache[b:b+1])
                if lf1_cache != None and lf2_cache != None:
                    lf1_cache[b:b+1].copy_(lf1)
                    lf2_cache[b:b+1].copy_(lf2)
                result.append(tmp_hidden_states.view(-1, *tmp_hidden_states.shape[2:]))
            hidden_states = torch.cat(result, dim=0)
        else:
            hidden_states = hidden_states.view(bsz, -1, hidden_size)
            hidden_states, lf1, lf2 = self.lf_gate(hidden_states, lf1_cache, lf2_cache)
            if lf1_cache != None and lf2_cache != None:
                lf1_cache.copy_(lf1)
                lf2_cache.copy_(lf2)
        hidden_states = hidden_states.contiguous().view(-1, hidden_states.shape[-1])
        qk, _ = self.qk_proj(hidden_states)
        qk = qk.view(*qk.shape[:-1], self.num_heads, int(qk.shape[-1] // self.num_heads))
        (q, k) = torch.chunk(qk, 2, dim=-1)
        
        q = apply_rotary_pos_emb(q , rotary_pos_emb, positions, use_yarn, yarn_scale_factor, attn_factor, attn_metadata)
        k = apply_rotary_pos_emb(k , rotary_pos_emb, positions, use_yarn, yarn_scale_factor, attn_factor, attn_metadata)
        v = v.view(*v.shape[:-2], self.num_heads * self.attn_head_size)
        q = q.view(-1, self.num_heads * self.attn_head_size)
        k = k.view(-1, self.num_heads * self.attn_head_size)
        
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class YuanDecoderLayer(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        linear_method: Optional[LinearMethodBase] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_projection_size = getattr(config, 'attention_projection_size', config.hidden_size)
        self.self_attn = YuanAttention(
            config=config,
            hidden_size=self.hidden_size,
            attention_projection_size=self.attention_projection_size,
            num_heads=config.num_attention_heads,
            linear_method=linear_method,
            num_kv_heads=config.num_attention_heads,
        )
        self.use_moe = getattr(config, "use_moe", False)
        if self.use_moe:
            self.mlp = YuanExperts(config, linear_method)
        else:
            self.mlp = YuanMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                    linear_method=linear_method,
            )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        kv_cache: torch.Tensor,
        lf1_cache: torch.Tensor,
        lf2_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        use_yarn: bool=False,
        yarn_scale_factor: float=1.0,
        attn_factor: float=1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            kv_cache=kv_cache,
            lf1_cache=lf1_cache,
            lf2_cache=lf2_cache,
            attn_metadata=attn_metadata,
            use_yarn=use_yarn,
            yarn_scale_factor=yarn_scale_factor,
            attn_factor=attn_factor
        )
        hidden_states = hidden_states.view(*residual.shape[:])
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, attn_metadata)
        hidden_states = residual + hidden_states

        return hidden_states


class YuanModel(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
 
        self.vocab_size = config.vocab_size + lora_vocab
        num_heads = getattr(config, "kv_channels", config.num_attention_heads)
        rotary_percent = getattr(config, "rotary_percent", 1.0)
        rotary_dim = int(config.hidden_size // num_heads * rotary_percent)
        self.use_yarn = getattr(config, "use_yarn", False)
        rotary_base = getattr(config, "rotary_base", 10000)
        self.yarn_scale_factor = getattr(config, "yarn_scale_factor", 128)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        self.attn_factor = getattr(config, "attn_factor", 1.0)
        scaled_max_position_embeddings = getattr(config, "scaled_max_position_embeddings", max_position_embeddings)
        torch_dtype = getattr(config, "torch_dtype", torch.bfloat16)

        if self.use_yarn:
            self.rotary_emb = YuanYaRNScaledRotaryEmbedding(rotary_dim, max_position_embeddings=scaled_max_position_embeddings, scale=self.yarn_scale_factor, original_max_position_embeddings=max_position_embeddings, attn_factor=self.attn_factor, dtype=torch_dtype)
            self.seq_len = scaled_max_position_embeddings 
        else:
            self.rotary_emb = YuanRotaryEmbedding(rotary_dim)
            self.seq_len = max_position_embeddings
        self.layers = nn.ModuleList([
            YuanDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        lf1_caches: List[torch.Tensor],
        lf2_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        rotary_pos_emb = self.rotary_emb(hidden_states, self.seq_len)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                positions,
                hidden_states,
                rotary_pos_emb,
                kv_caches[i],
                lf1_caches[i],
                lf2_caches[i],
                attn_metadata,
                self.use_yarn,
                self.yarn_scale_factor,
                self.attn_factor
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class YuanForCausalLM(nn.Module):

    def __init__(
        self,
        config: YuanConfig,
        linear_method: Optional[LinearMethodBase] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.use_moe = getattr(config, "use_moe", False)
        self.linear_method = linear_method
        self.model = YuanModel(config, linear_method, lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            if not lora_config else lora_config.lora_vocab_padding_size,
        )

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        lf1_caches: List[torch.Tensor],
        lf2_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if attn_metadata.prefill_metadata == None:
            bsz = attn_metadata.num_decode_tokens
        else:
            bsz = attn_metadata.num_prefills
        #bsz = 1
        hidden_states = self.model(input_ids, positions, kv_caches, lf1_caches, lf2_caches, attn_metadata)
        return hidden_states 

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        q_projs, k_projs= {}, {}

        if self.use_moe:
            moe_state_dict = {}
        for name, loaded_weight in weights:
            if "rotary_emb" in name:
                continue
            if 'q_proj' in name:
                q_projs[name] = loaded_weight
                continue
            if 'k_proj' in name:
                k_projs[name] = loaded_weight
                continue
            if self.use_moe:
                if 'mlp' in name:
                    moe_state_dict[name] = loaded_weight
                    continue
            param = params_dict[name]
            if name.endswith(".bias") and name not in params_dict:
                continue
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            tp_rank = get_tensor_model_parallel_rank()
        for i in range(self.config.num_hidden_layers):
            name = f'model.layers.{i}.self_attn.qk_proj.weight'
            q_name = f'model.layers.{i}.self_attn.q_proj.weight'
            k_name = f'model.layers.{i}.self_attn.k_proj.weight'
            qk_weight = torch.cat([q_projs[q_name], k_projs[k_name]], dim=0)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, qk_weight)
        if self.use_moe:
            for layer_id in range(self.config.num_hidden_layers):
                qkv = []
                for key in ['query', 'key', 'value']:
                    hf_name = f'model.layers.{layer_id}.mlp.gate.{key}.weight'
                    qkv.append(moe_state_dict[hf_name])
                qkv_weight = torch.cat(qkv, dim=0)
                
                name = f'model.layers.{layer_id}.mlp.router.query_key_value.weight'
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, qkv_weight)

                experts = []
                for expert_id in range(self.config.moe_config['moe_num_experts']):
                    hf_name = f'model.layers.{layer_id}.mlp.experts.{expert_id}.w1.weight'
                    experts.append(moe_state_dict[hf_name].T.unsqueeze(0)) #.view(1, *moe_state_dict[hf_name].shape))
                experts_weight = torch.cat(experts, dim=0)
                name = f'model.layers.{layer_id}.mlp.ws'
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, experts_weight, name)

                experts = []
                for expert_id in range(self.config.moe_config['moe_num_experts']):
                    hf_name = f'model.layers.{layer_id}.mlp.experts.{expert_id}.w2.weight'
                    experts.append(moe_state_dict[hf_name].T.unsqueeze(0)) #view(1, *moe_state_dict[hf_name].shape))
                experts_weight = torch.cat(experts, dim=0)
                name = f'model.layers.{layer_id}.mlp.w2s'
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, experts_weight, name)
