# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Split tensor parallel partitions."""

import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

from megatron.checkpointing import load_checkpoint, save_checkpoint, _load_base_checkpoint, get_distributed_optimizer_checkpoint_name
from megatron.checkpointing import ensure_directory_exists
from megatron.checkpointing import get_checkpoint_name
from megatron.checkpointing import get_checkpoint_version
from megatron.checkpointing import get_checkpoint_tracker_filename
from megatron.global_vars import set_global_variables, get_args
from megatron.global_vars import rebuild_tokenizer
from megatron.initialize import initialize_megatron
from megatron.arguments import (parse_args, validate_args)
from megatron.core import mpu
from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.global_vars import get_args
from megatron.utils import (unwrap_model, print_rank_0)
from megatron.checkpointing import _load_base_checkpoint
from megatron.optimizer import get_megatron_optimizer, get_param_groups
from megatron.training import  get_optimizer_param_scheduler
from megatron.checkpointing import load_checkpoint
from copy import deepcopy
from tqdm import tqdm
from pretrain_yuan import model_provider

def get_model():
    args = get_args()
    
    pre_process = True if mpu.is_pipeline_first_stage() else False
    post_process = True if mpu.is_pipeline_last_stage() else False
    model = model_provider(pre_process, post_process)
    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    model = [LocalDDP(model_module,
                      args.accumulate_allreduce_grads_in_fp32,
                      args.use_contiguous_buffers_in_local_ddp)
             for model_module in model]
    return model

def get_parallel_checkpoint_name(path):

    tracker_filename = get_checkpoint_tracker_filename(path)
    iteration = 0
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        iteration = int(metastring)
    assert iteration > 0
    checkpoint_name = get_checkpoint_name(path, iteration)

    return checkpoint_name, iteration

def get_mp_merge_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='mp merge')

    group.add_argument('--model-type', type=str,help='Type of the model.')
    group.add_argument('--target-tensor-model-parallel-size', type=int, default=2,
                       help='Degree of pipeline model parallelism in output model.')
    group.add_argument('--target-pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism in output model.')
    group.add_argument('--with-distributed-optimizer', action='store_true',
                       help='Use distributed optimizer during split ckpt.')
    group.add_argument('--pipeline-generate-layer', type=str, default=None,help='This parameter controls which layers only convert the paramater.')
    group.add_argument('--tensor-generate-layer', type=str, default=None, help='THis parameter controls which layers only convert the parameter.')
    return parser



def main():
    # Arguments do sanity checks on the world size, but we don't care,
    # so trick it into thinking we are plenty of processes
    os.environ["WORLD_SIZE"] = "{}".format(2**8)

    # Args
    args = parse_args(extra_args_provider=get_mp_merge_args, ignore_unknown_args=True)
    validate_args(args)
    set_global_variables(args)
    args = get_args()
    args.model_type=ModelType.encoder_or_decoder
    args.orig_tensor_model_parallel_size = args.tensor_model_parallel_size
    args.orig_pipeline_model_parallel_size = args.pipeline_model_parallel_size
    args.orig_transformer_pipeline_model_parallel_size = args.transformer_pipeline_model_parallel_size

    args.target_transformer_pipeline_model_parallel_size = (
        args.target_pipeline_model_parallel_size - 1
        if args.standalone_embedding_stage else
        args.target_pipeline_model_parallel_size
    )
    #tokenizer = rebuild_tokenizer(args)

    print('\n spliting tensor parallel partitions ...')
    print(' > orig number of partitions: {}'.format(args.orig_tensor_model_parallel_size))
    print(' > checkpoint path: {}'.format(args.load))
    print(' > model parameters:')
    print('    number of layers ................ {}'.format(args.num_layers))
    print('    hidden size ..................... {}'.format(args.hidden_size))
    print('    number of attention heads ....... {}'.format(args.num_attention_heads))
    if args.position_embedding_type != 'rope':
        print('    maximum position embeddings ..... {}'.format(args.max_position_embeddings))

    # Build and load partitions.
    partitions = []
    tokenizer = rebuild_tokenizer(args)
    pipeline_generate_layer_index = [int(x) for x in args.pipeline_generate_layer.split(',')]
    sub_tensor_parallel_size = args.target_tensor_model_parallel_size // args.orig_tensor_model_parallel_size
    for pp_rank in pipeline_generate_layer_index:
        for tp_rank in range(args.orig_tensor_model_parallel_size):
            print('processing pp_rank {}, tp_rank {}'.format(pp_rank,tp_rank))
            # set orig pp_rank and tp_rank
            args.tensor_model_parallel_size = args.orig_tensor_model_parallel_size
            args.pipeline_model_parallel_size = args.orig_pipeline_model_parallel_size
            args.transformer_pipeline_model_parallel_size = args.orig_transformer_pipeline_model_parallel_size
            mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
            mpu.set_tensor_model_parallel_rank(tp_rank)
            mpu.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
            mpu.set_pipeline_model_parallel_rank(pp_rank)
            # build orig model
            model_ = get_model()
            model = unwrap_model(model_)
            state_dict, checkpoint_name, release = _load_base_checkpoint(args.load, rank0=False)

            # Load orig Model.
            if len(model) == 1:
                model[0].load_state_dict(state_dict['model'], strict=True)
            else:
                for i in range(len(model)):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    model[i].load_state_dict(state_dict['model%d' % i], strict=True)
            total_numel = 0
            for name, param in model[0].named_parameters():
                total_numel += param.numel()
    
            if not args.no_load_optim and args.use_distributed_optimizer:
                optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(checkpoint_name)
                optim_state_dict = torch.load(optim_checkpoint_name, map_location='cpu')
                assert total_numel == optim_state_dict[0][torch.float32]['param'].shape[0]
            # build param_groups for optimizer
            param_groups = get_param_groups(model_, None, None, 1.0)

            # the model structure of each tp is the same 
            args.tensor_model_parallel_size = args.target_tensor_model_parallel_size
            args.pipeline_model_parallel_size = args.target_pipeline_model_parallel_size
            args.transformer_pipeline_model_parallel_size = args.target_transformer_pipeline_model_parallel_size
            mpu.set_tensor_model_parallel_world_size(args.target_tensor_model_parallel_size)
            mpu.set_tensor_model_parallel_rank(tp_rank * sub_tensor_parallel_size)
            mpu.set_pipeline_model_parallel_world_size(args.pipeline_model_parallel_size)
            mpu.set_pipeline_model_parallel_rank(pp_rank)

            sub_model_ = get_model()
            sub_param_groups = get_param_groups(sub_model_, None, None, 1.0)
            prefix = 'module.module.language_model'

            for sub_tp_rank in range(sub_tensor_parallel_size):
                # only modify tensor parallel rank
                mpu.set_tensor_model_parallel_rank(tp_rank * sub_tensor_parallel_size + sub_tp_rank)
                # modify weight in sub_state_dict
                sub_state_dict = deepcopy(state_dict)
                for (name, param), (sub_name, sub_param) in zip(model_[0].named_parameters(), sub_model_[0].named_parameters()):
                    if param.tensor_model_parallel:
                        if 'mlp.experts.weight1' in sub_name:
                            param.data = param.data.view(args.num_experts, args.hidden_size, -1)
                            sub_param.data = sub_param.data.view(args.num_experts, args.hidden_size, -1)
                            chunk_size = param.shape[param.partition_dim+1] // 2
                            chunk0 = torch.split(param.data, chunk_size, dim=(param.partition_dim+1))[0].clone().detach()
                            chunk1 = torch.split(param.data, chunk_size, dim=(param.partition_dim+1))[1].clone().detach()
                            sub_chunk_size = sub_param.shape[param.partition_dim+1] // 2
                            sub_chunk0 = torch.split(chunk0, sub_chunk_size, dim=(param.partition_dim+1))[sub_tp_rank].clone().detach()
                            sub_chunk1 = torch.split(chunk1, sub_chunk_size, dim=(param.partition_dim+1))[sub_tp_rank].clone().detach()
                            sub_param.data.copy_(torch.cat([sub_chunk0, sub_chunk1], dim=(param.partition_dim+1)))
                            sub_param.data = sub_param.data.view(args.hidden_size,-1)
                        elif 'mlp.experts.weight2' in sub_name:
                            param.data = param.data.view(args.num_experts, -1, args.hidden_size)
                            sub_param.data = sub_param.data.view(args.num_experts, -1, args.hidden_size)
                            sub_chunk_size = sub_param.shape[param.partition_dim+1]
                            sub_param.data.copy_(torch.split(param.data, sub_chunk_size, dim=(param.partition_dim+1))[sub_tp_rank].clone().detach())
                            sub_param.data = sub_param.data.view(-1,args.hidden_size)
                        elif 'dense_h_to_4h' in sub_name:
                            chunk_size = param.shape[param.partition_dim]//2
                            chunk0 = torch.split(param.data, chunk_size, dim=param.partition_dim)[0].clone().detach()
                            chunk1 = torch.split(param.data, chunk_size, dim=param.partition_dim)[1].clone().detach()
                            chunk_size = sub_param.shape[param.partition_dim]//2
                            chunk0 = torch.split(chunk0, chunk_size, dim=param.partition_dim)[sub_tp_rank].clone().detach()
                            chunk1 = torch.split(chunk1, chunk_size, dim=param.partition_dim)[sub_tp_rank].clone().detach()
                            sub_param.data.copy_(torch.cat([chunk0, chunk1], dim=param.partition_dim))
                        else:
                            chunk_size = sub_param.shape[param.partition_dim]
                            sub_param.data.copy_(torch.split(param.data, chunk_size, dim=param.partition_dim)[sub_tp_rank].clone().detach())
                    else:
                        sub_param.data.copy_(param.data.clone().detach())
                sub_model = unwrap_model(sub_model_)
                sub_state_dict['model'] = sub_model[0].state_dict_for_save_checkpoint()

                sub_state_dict['args'].tensor_model_parallel_size = args.target_tensor_model_parallel_size
                # output state dict ckpt file
                iteration = state_dict['iteration']
                sub_checkpoint_name = get_checkpoint_name(args.save, iteration)
                ensure_directory_exists(sub_checkpoint_name)
                print('saving to ', sub_checkpoint_name)
                torch.save(sub_state_dict, sub_checkpoint_name)
                # writing txt file
                if not torch.distributed.is_initialized() \
                   or torch.distributed.get_rank() == 0:
                    tracker_filename = get_checkpoint_tracker_filename(args.save)
                    with open(tracker_filename, 'w') as f:
                        f.write(str(iteration))
    print('done :-)')


if __name__ == '__main__':
    main()
