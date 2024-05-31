import sys
import os
import torch
from abc import ABC
from tqdm import tqdm
from torch.utils.data import Dataset

sys.path.append('./')
from megatron import get_args
from megatron.core import mpu
from megatron import get_tokenizer
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process


def model_provider(pre_process=True, post_process=True):
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)
    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument('--max_len', type=int, default=1024)
    group.add_argument('--model_config_path', type=str, default='./')
    group.add_argument('--math_datapath', type=str, default='./')
    group.add_argument('--output_path', type=str, default='./')
    group.add_argument('--num_samples_per_task', type=int, default=10)
    group.add_argument('--top_k', type=int, default=0)
    group.add_argument('--top_p', type=float, default=0.95)
    group.add_argument('--top_p_decay', type=float, default=0.0)
    group.add_argument('--top_p_bound', type=float, default=0.0)
    group.add_argument('--temp', type=float, default=0.5)
    group.add_argument('--min_length', type=int, default=0)
    group.add_argument('--random_seed', type=int, default=1234)
    group.add_argument('--beam_width', type=int, default=None)
    group.add_argument('--length_penalty', type=int, default=1)
    group.add_argument('--prevent_newline_after_colon', type=bool, default=False)
    return parser

def clean_tab(msg_text):
    __sep_note = "<n>"
    msg_text = msg_text.replace("\n", __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    msg_text = msg_text.replace(__sep_note + __sep_note, __sep_note)
    return msg_text

class EvalDataset(ABC, Dataset):
    def __init__(self, data_path):
        self.problems = []
        self.keys = []
        self.answers = []

        with open(data_path, 'r') as f:
            lines = f.readlines()
            for ii, line in enumerate(lines):
                line = line.strip()
                index = line.find('[SEP]')
                prompt = '''Question:What is the diagonal length of a square field with an area of 50 square meters?<n>Answer:To find the diagonal length of a square field, we first need to find the length of one side of the square. Since the area of the square is given as 50 square meters, we can use the formula for the area of a square:<n>Area = side^2<n>Let's call the length of one side of the square "s". Then we have:<n>50 = s^2<n>To find the length of one side, we take the square root of both sides:<n>s = √50<n>s = √(25 * 2)<n>s = √25 * √2<n>s = 5√2 meters<n>Now that we have the length of one side, we can find the diagonal using the Pythagorean theorem. In a square, the diagonal forms a right-angled triangle with two sides of the square. Let's call the diagonal "d". Then we have:<n>d^2 = s^2 + s^2<n>d^2 = (5√2)^2 + (5√2)^2<n>d^2 = 2 * (5√2)^2<n>d^2 = 2 * (5^2 * 2)<n>d^2 = 2 * (25 * 2)<n>d^2 = 2 * 50<n>d^2 = 100<n>Now take the square root of both sides to find the diagonal:<n>d = √100<n>d = 10 meters<n>So, the diagonal length of the square field is 10 meters.<n>Question:A rectangular fish tank measures 10 cm in length, 5 cm in width, and 6 cm in height. How many such fish tanks will it take to fill a cubic container with side length 20 cm?<n>To solve this problem, we'll first find the volume of one rectangular fish tank and then the volume of the cubic container. Finally, we'll divide the volume of the cubic container by the volume of one fish tank to determine how many fish tanks are needed to fill it.<n>Answer:To solve this problem, we need to find the volume of one rectangular fish tank and the volume of the cubic container. Then we can divide the volume of the cubic container by the volume of one fish tank to find how many fish tanks it will take to fill the container.<n>First, let's find the volume of one rectangular fish tank:<n>V_tank = lwh<n>V_tank = (10 cm)(5 cm)(6 cm)<n>V_tank = 300 cm^3<n>Next, let's find the volume of the cubic container:<n>V_container = s^3<n>V_container = (20 cm)^3<n>V_container = 8000 cm^3<n>Now, we can divide the volume of the cubic container by the volume of one fish tank to find how many fish tanks it will take to fill the container:<n>Number of fish tanks = V_container / V_tank<n>Number of fish tanks = (8000 cm^3) / (300 cm^3)<n>Number of fish tanks = 26.67<n>Since we cannot have a fraction of a fish tank, we must round up to the nearest whole number. Therefore, it will take 27 fish tanks to fill the cubic container.<n>Answer: It will take 27 such fish tanks to fill the cubic container with side length 20 cm.<n>Question:Find the smallest real number x in the domain of the function f(x) = \\frac {x^2 - 4}{x - 3} such that f(x) is a real number.<n>Answer:To determine the domain of the function f(x), we need to consider any values of x that would result in division by zero. In this case, division by zero occurs when the denominator, x - 3, is equal to zero. Therefore, we need to find the value(s) of x that satisfy x - 3 = 0.<n>Solving for x, we have:<n>x - 3 = 0<n>x = 3<n>So, x = 3 is the value that makes the denominator zero. However, we want to find the smallest real number x in the domain such that f(x) is a real number.<n>To determine whether f(x) is a real number or not, we need to check if the numerator, x^2 - 4, is also zero when x = 3. Substituting x = 3 into the numerator, we get:<n>f(3) = \\frac {3^2 - 4}{3 - 3} = \\frac {9 - 4}{0} = \\frac {5}{0}<n>The expression \\frac {5}{0} is undefined because division by zero is not allowed in mathematics. Therefore, f(x) is not a real number when x = 3.<n>Since f(x) is not a real number at x = 3, we need to find the next smallest real number in the domain. As we move to the left on the number line, the function f(x) approaches negative infinity. Hence, there are no real numbers smaller than 3 in the domain of f(x).<n>Therefore, there is no smallest real number x in the domain of f(x) such that f(x) is a real number.<n>Question:Simplify (3x - 5) + (2 - 4x).<n>Answer:To simplify the expression (3x - 5) + (2 - 4x), we can follow the same steps as before:<n>Step 1: Distribute the positive sign to each term inside the second set of parentheses:<n>(3x - 5) + (2 - 4x) = 3x - 5 + 2 - 4x<n>Step 2: Rearrange the terms so that like terms are next to each other:<n>3x - 4x - 5 + 2<n>Step 3: Combine the like terms:<n>(3x - 4x) + (-5 + 2) = -x - 3<n>The simplified expression is -x - 3.<n>Question:'''
                line = prompt + line[:index] + '<sep>'
                line = line.replace('<n>', '\n')
                self.problems.append(line)
                self.keys.append(ii)
                self.answers.append('')

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        try:
            key = self.keys[idx]
            sample = self.problems[key]
        except Exception as e:
            print(e, idx, len(self.problems))
            exit()
        return {'task_id':key, 'sample':sample}


def main():
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'YuanTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()
    dataset = EvalDataset(args.math_datapath)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=mpu.get_data_parallel_rank(), num_replicas = mpu.get_data_parallel_world_size(), shuffle=False, drop_last=False)
    data_loader = torch.utils.data.DataLoader(dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=2)
    model = get_model(model_provider, wrap_with_ddp=False)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    tokenizer = get_tokenizer()
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False
    tokenizer.eod = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    stop_token = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    torch.distributed.barrier()
    model.eval()
    if args.fp16:
        model = model.half()
    elif args.bf16:
        model = model.bfloat16()
    else:
        model = model.float()
    model.cuda()
    torch.distributed.barrier()
    if torch.distributed.get_rank()==0 and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with torch.no_grad():
        data_iter = tqdm(enumerate(data_loader), total=len(data_loader)) if torch.distributed.get_rank()==0 else enumerate(data_loader)
        for i, batch in data_iter:
            sample_iter = tqdm(range(args.num_samples_per_task), total=args.num_samples_per_task) if torch.distributed.get_rank()==0 else  range(args.num_samples_per_task)
            for j in sample_iter:
                def inference_once(top_k=None, top_p=None, temp=None, seed=None):
                    tokens = tokenizer(batch['sample'], return_tensors='pt', padding=True).input_ids[:,:-1].to(torch.cuda.current_device())
                    if args.beam_width is not None:
                        response, response_seg, response_scores = \
                            beam_search_and_post_process(
                            model,
                            prompts=batch['sample'],
                            tokens_to_generate=(args.max_len - len(tokens)),
                            beam_size = args.beam_width,
                            add_BOS=False,
                            stop_token=stop_token,
                            num_return_gen=args.beam_width,
                            length_penalty=args.length_penalty,
                            prevent_newline_after_colon=args.prevent_newline_after_colon
                            )
                    else:
                        response, response_seg, response_logprobs, _ = \
                            generate_and_post_process(
                            model,
                            prompts=batch['sample'],
                            tokens_to_generate=(args.max_len - len(tokens)),
                            return_output_log_probs=False,
                            top_k_sampling=top_k,
                            top_p_sampling=top_p,
                            top_p_decay=args.top_p_decay,
                            top_p_bound=args.top_p_bound,
                            temperature=temp,
                            add_BOS=False,
                            stop_on_eol=False,
                            prevent_newline_after_colon=args.prevent_newline_after_colon,
                            random_seed=seed)

                    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
                        if response[0][0]==' ':
                            response = [response[0][1:-5]]
                        else:
                            response = [response[0][0:-5]]
                        new_sample = response
                        print('\n' + response[0])

                        with open(os.path.join(args.output_path, f'samples_{args.rank}.jsonl'), 'a', encoding='utf-8') as fp:
                            for _, x in enumerate(new_sample):
                                res = x.strip()
                                res = res.replace('<pad>', '')
                                res = res.replace('<eod>', '')
                                res = res.replace('<sep>', '[SEP]')
                                res = clean_tab(res)
                                record_res = res.strip() + '\n'
                                fp.write(record_res)
                inference_once(top_k=args.top_k, top_p=args.top_p, temp=args.temp, seed=args.random_seed)
              
    torch.distributed.barrier()


if __name__ == '__main__':
    main()
