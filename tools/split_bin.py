import torch,transformers
import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from transformers import AutoModelForCausalLM,AutoConfig, LlamaTokenizer,AutoTokenizer
from yuan_moe_hf_model import YuanForCausalLM

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input-path', type=str, help='Path to the input file')
parser.add_argument('--output-path', type=str, help='Path to the output file')

args = parser.parse_args()

model = YuanForCausalLM.from_pretrained(args.input_path,device_map='auto',torch_dtype=torch.bfloat16,trust_remote_code=True)

model.save_pretrained(args.output_path,max_shard_size='10GB')

