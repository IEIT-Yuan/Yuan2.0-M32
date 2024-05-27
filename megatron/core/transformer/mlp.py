# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Tuple, Union

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron.core import tensor_parallel
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
# from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


@dataclass
class MLPSubmodules:
    linear_fc1: Union[ModuleSpec, type] = None
    linear_fc2: Union[ModuleSpec, type] = None


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules, is_expert: bool = False
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        ffn_hidden_size = self.config.ffn_hidden_size
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        self.linear_fc1 = TEColumnParallelLinear(
            self.config.hidden_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
        )

        if self.config.gated_linear_unit:
            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]
            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = TERowParallelLinear(
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
        )

    def forward(self, hidden_states):

        # [s, b, 4 * h/p]
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        if self.config.bias_gelu_fusion:
            assert self.config.add_bias_linear is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        output, output_bias = self.linear_fc2(intermediate_parallel)
        return output, output_bias

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config):
        super(ParallelMLP, self).__init__(config=config)
        args = get_args()
        self.add_bias = config.add_bias_linear
        if args.num_shared_experts is not None:
            ffn_hidden_size = config.ffn_hidden_size * args.num_shared_experts 
        self.moe_intermediate_size = ffn_hidden_size
        if config.gated_linear_unit:
            ffn_hidden_size *= 2
        self.num_shared_experts = args.num_shared_experts
        self.isolate_shared_experts = args.isolate_shared_experts
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
        )

        self.bias_gelu_fusion = False
        self.activation_func = None
        self.swiglu = args.swiglu

        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        elif args.swiglu:
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu
        elif args.squared_relu:
            def squared_relu(x):
                return torch.pow(F.relu(x), 2)
            self.activation_func = squared_relu
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu
        # Project back to h.
        self.shared_experts = torch.nn.ModuleList()
        if self.isolate_shared_experts:
            for i in range(self.num_shared_experts):
                self.shared_experts.append(tensor_parallel.RowParallelLinear(
                                            config.ffn_hidden_size,
                                            config.hidden_size,
                                            config=config,
                                            init_method=config.output_layer_init_method,
                                            bias=self.add_bias,
                                            input_is_parallel=True)
                                            )
        else:
            self.shared_experts.append(tensor_parallel.RowParallelLinear(
                                            self.moe_intermediate_size,
                                            config.hidden_size,
                                            config=config,
                                            init_method=config.output_layer_init_method,
                                            bias=self.add_bias,
                                            input_is_parallel=True)
                                          )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
            assert self.add_bias is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h]
        if self.isolate_shared_experts:  
            intermediate_parallel = torch.chunk(intermediate_parallel, self.num_shared_experts, dim=-1)
            output = torch.zeros_like(hidden_states)
            for expert_num, expert in enumerate(self.shared_experts):
                output += expert(intermediate_parallel[expert_num])[0]
        else:
            for expert_num, expert in enumerate(self.shared_experts):
                output = expert(intermediate_parallel)[0]

        return output, None

