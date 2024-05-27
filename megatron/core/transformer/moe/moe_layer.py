# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch
from megatron import get_args
from megatron.core import parallel_state
from megatron.core.transformer.mlp import MLPSubmodules,ParallelMLP
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import MoEDroplessTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = 1
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"
        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            0
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.token_dispatcher = None

    @abstractmethod
    def forward(self, hidden_states):
        pass


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(self, config: TransformerConfig, submodules: MLPSubmodules = None):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config)
        args = get_args()
        self.use_fp32_router = args.use_fp32_router
        self.router = TopKRouter(config=self.config)
        if args.num_shared_experts is not None:
            self.mlp = ParallelMLP(config)
        self.num_shared_experts = args.num_shared_experts
        if self.config.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        self.token_dispatcher = MoEDroplessTokenDispatcher(
            self.num_local_experts, self.local_expert_indices, config=self.config
        )

    def forward(self, hidden_states: torch.Tensor):
        # process MoE
        # hidden_states: [SeqLen/TP, MBS, hidden_states]
        # scores, indices: [SeqLen/TP * MBS, num_moe_experts]
        scores, indices = self.router(hidden_states)
        if self.use_fp32_router:
            scores = scores.to(hidden_states.dtype)
        (
            dispatched_input,
            tokens_per_expert,
            scores,
            indices,
            global_local_map,
        ) = self.token_dispatcher.token_permutation(hidden_states, scores, indices)
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        output, mlp_bias = self.token_dispatcher.token_unpermutation(
            expert_output, scores, indices, global_local_map, mlp_bias
        )
        if self.num_shared_experts is not None:
            output = output + self.mlp(hidden_states)[0]
        return output, mlp_bias
