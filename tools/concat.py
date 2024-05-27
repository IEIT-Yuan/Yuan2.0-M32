import argparse
import torch
import os
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--input-path', type=str, help='Path to the input file')
parser.add_argument('--output-path', type=str, help='Path to the output file')
parser.add_argument('--pp-rank', type=int, default=8, help='Path to the input file')
parser.add_argument('--num-layers', type=str,  default=24, help='Path to the output file')

args = parser.parse_args()

bin_path = args.input_path
bin_name = 'pytorch_model_'


save_path = os.path.join(args.output_path, 'pytorch_model.bin') 

pp_rank = args.pp_rank
num_layers = args.num_layers

layers_per_rank = num_layers//pp_rank


model_list = []

for i in range(pp_rank):
    load_file_path = os.path.join(bin_path, bin_name+str(i)+'.bin')
    model = torch.load(load_file_path)
    print('model no.', i, ' key number:', len(model.keys()))
    model_list.append(model)    

new_state_dict = model_list[0]

current_layer_no = layers_per_rank

for j in range(pp_rank-1):
    model = model_list[j+1]
    
    for layer_no in range(layers_per_rank):
        keys_in_one_block = []
        for layer_key in model.keys():
            if 'layers.'+str(layer_no)+'.' in layer_key:
                keys_in_one_block.append(layer_key)
        
        for layer_key in keys_in_one_block:
            new_layer_key = layer_key.replace('layers.'+str(layer_no), 'layers.'+str(current_layer_no))
            new_state_dict[new_layer_key] = model[layer_key]
        
        current_layer_no += 1

new_state_dict['model.norm.weight'] = model_list[-1]['model.norm.weight']
new_state_dict['lm_head.weight'] = model_list[-1]['lm_head.weight']

torch.save(new_state_dict, save_path)

