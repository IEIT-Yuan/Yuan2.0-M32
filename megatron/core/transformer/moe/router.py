# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math
from abc import ABC, abstractmethod
from typing import Callable, List

import torch

from megatron import get_timers, get_args, get_retro_args, core, get_num_microbatches
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    sinkhorn,
    switch_load_balancing_loss_func,
    z_loss_func,
)
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron import get_args
from megatron.core import mpu, tensor_parallel
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu

try:
    from einops import rearrange
except ImportError:
    rearrange = None

try:
    from flash_attn import flash_attn_varlen_func as flash_attn_unpadded_func
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_unpadded_func = None



class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                    device=q.device)

        if self.training:
            # during training q,k,v always have same seqlen
            assert seqlen_k == seqlen_q

            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                        device=q.device)
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
        return output


class CoreAttention(MegatronModule):

    def __init__(self, layer_number, config,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__(config)
        self.fp16 = config.fp16
        self.bf16 = config.bf16
        args  = get_args()
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = config.sequence_parallel
        projection_size = args.seq_length 
        
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition =  projection_size
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_router_heads)
        self.num_attention_heads_per_partition = config.num_attention_heads 

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            config.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        if self.training == False:
            self.hidden_size_per_partition = query_layer.size(2)
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class ParallelAttention_router(MegatronModule):
    def __init__(self, config, layer_number=0,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention_router, self).__init__(config)
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = config.params_dtype
        self.sequence_parallel = config.sequence_parallel
        self.flash_attn_drop = args.flash_attn_drop
        self.use_lf_gate = args.use_lf_gate
        self.hidden_size = config.hidden_size

        self.use_flash_attn = args.use_flash_attn
        self.use_fp32_router = args.use_fp32_router

        projection_size = args.num_experts

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
           args.seq_length , args.num_attention_router_heads)
        self.num_attention_router_heads = args.num_attention_router_heads
        self.num_attention_heads_per_partition = config.num_attention_heads

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
                self.query_key_value = tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    3 * projection_size,
                    config=config,
                    init_method=config.init_method,
                    bias=args.add_bias_linear,
                    gather_output=True)

        self.core_attention = CoreAttention(self.layer_number, config,
                                            AttnMaskType.padding)
        self.checkpoint_core_attention = config.recompute_granularity == 'selective'


    def forward(self, hidden_states, attention_mask=None, enc_position_ids=None,
                encoder_output=None, inference_params=None,
                rotary_pos_emb=None):
        is_first_step = False
        before_hidden_states = None
        if self.attention_type == AttnType.self_attn:
            mixed_x_layer, _ = self.query_key_value(hidden_states)        
            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)

        seq_length = query_layer.size(0)
        batch_size = query_layer.size(1)
        expert_num = query_layer.size(2)

        if self.training == False:
            self.num_attention_router_heads = seq_length // self.hidden_size_per_attention_head

        query_layer = query_layer.transpose(0, 2).contiguous().view(expert_num, batch_size, self.num_attention_router_heads, self.hidden_size_per_attention_head)
        key_layer = key_layer.transpose(0, 2).contiguous().view(expert_num, batch_size, self.num_attention_router_heads, self.hidden_size_per_attention_head)
        value_layer = value_layer.transpose(0, 2).contiguous().view(expert_num, batch_size, self.num_attention_router_heads, self.hidden_size_per_attention_head)
        
        if self.use_fp32_router:
            context_layer = self.core_attention(
                query_layer.float(), key_layer.float(), value_layer.float(), None)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, None)
        
        router_output = context_layer.transpose(0, 2).contiguous()
        return router_output

class Router(ABC, MegatronModule):
    """Base Router class"""

    def __init__(self, config: TransformerConfig) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__(config)
        self.config = config
        args = get_args()
        self.use_attention_router = args.use_attention_router
        self.num_moe_experts = self.config.num_moe_experts
        self.moe_aux_loss_func = None
        if self.use_attention_router:
            self.attention_router = ParallelAttention_router(config)
        else:
            self.weight = torch.nn.Parameter(
                torch.empty((self.config.num_moe_experts, self.config.hidden_size))
            )
            if args.process_checkpoint:
                config.init_method(self.weight)
            else:
                with get_cuda_rng_tracker().fork():
                    config.init_method(self.weight)
            setattr(self.weight, 'sequence_parallel', config.sequence_parallel)
        

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor. [SeqLen/TP, MBS, HiddenSize]

        Returns:
            torch.Tensor: Logits tensor.
        """
        # logits: [SeqLen/TP, MBS, num_moe_experts]
        if self.use_attention_router:
            logits = self.attention_router(input)
        else:
            logits = torch.nn.functional.linear(input, self.weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors representing max probs and the indices.
        """
        raise NotImplementedError("Routing function not implemented.")

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor. [SeqLen/TP, MBS, HiddenSize]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: scores and indices.
        """
        # self.hidden [SeqLen/TP, MBS, HiddenSize]
        self.hidden = input.shape[-1]
        # logits [SeqLen/TP, MBS, num_moe_experts]
        logits = self.gating(input)
        # logits [SeqLen/TP * MBS, num_moe_experts]
        logits = logits.view(-1, self.config.num_moe_experts)
        
        scores, indices = self.routing(logits)

        return scores, indices


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(self, config: TransformerConfig,) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(config=config)
        assert config.moe_token_dropping is False
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type
        self.moe_aux_loss_func = switch_load_balancing_loss_func
        self.input_jitter = None

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.topk == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.topk, dim=1)
        return scores, indices

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The scores and the indices tensor after applying load balancing.
        """
        # 取topk，top_logits, indices [SeqLen/TP * MBS, TopK]
        top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
        # scores [SeqLen/TP * MBS, TopK]
        scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        # Apply load balancing loss
        # probs [SeqLen/TP * MBS, num_moe_experts]
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
        # scores: [SeqLen/TP * MBS, num_moe_experts]
        scores = self.apply_aux_loss(self.moe_aux_loss_func, probs, indices, activation=scores)
        return scores, indices

    def apply_aux_loss(
        self,
        loss_func: Callable,
        probs: torch.Tensor,
        indices: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            loss_func (callable): The loss function to be used. switch_load_balancing_loss_func
            probs (torch.Tensor): The probabilities output by the MoE layer. [SeqLen/TP * MBS, num_moe_experts]
            indices (torch.Tensor): The indices of the selected experts. [SeqLen/TP * MBS, TopK]
            activation (torch.Tensor): The activation tensor to attach the gradient function to. [SeqLen/TP * MBS, TopK]

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        mask = torch.nn.functional.one_hot(indices, num_classes=self.num_moe_experts).sum(dim=1)
        aux_loss = loss_func(probs, mask, self.config.moe_aux_loss_coeff)
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.
        
        Args:
            logits (torch.Tensor): The logits of the router.
        
        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.moe_z_loss_coeff is not None:
            z_loss = z_loss_func(logits, self.config.moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.moe_input_jitter_eps is not None:
            eps = self.config.moe_input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample
            return input * self.input_jitter(input.shape)
        else:
            return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor. [SeqLen/TP * MBS, num_moe_experts]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
        """
        logits = logits.view(-1, self.config.num_moe_experts)

        # Apply Z-Loss, ST-MOE
        # L_z(x) = \frac{1}{B} \sum_{i=1}^B \left(  log \sum_{j=1}^N e^{x_j^{(i)}} \right)^2
        logits = self.apply_z_loss(logits)
        # Apply input jitter ST-MOE
        logits = self.apply_input_jitter(logits)

        if self.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.routing_type == "none":
            # A naive top-k routing without load balancing
            # top_logits, indices [SeqLen/TP * MBS, TopK]
            top_logits, indices = torch.topk(logits, k=self.topk, dim=1)
            scores = torch.softmax(top_logits, dim=-1, dtype=torch.float32).type_as(logits)
        else:
            raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
        # scores, indices: [SeqLen/TP * MBS, num_moe_experts]
        return scores, indices
