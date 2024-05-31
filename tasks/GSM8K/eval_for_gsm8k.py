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
                prompt = '''Question:On Tuesday, Mike had 45 books and Corey had twice as many books as Mike. On Wednesday, Mike gave Lily 30% of his books, while Corey gave Lily 5 more books than what Mike gave to Lily. In the evening, Lily's friend, Emma, gave Lily 1/4 of her 28 books collection. How many books did Lily get on Wednesday?<n>Answer:First, let's find out how many books Corey had on Tuesday. Since Corey had twice as many books as Mike, and Mike had 45 books:<n>Corey's books = 2 * Mike's books<n>Corey's books = 2 * 45<n>Corey's books = 90 books<n>On Wednesday, Mike gave Lily 30% of his books:<n>Mike's books given to Lily = 30% of 45<n>Mike's books given to Lily = 0.30 * 45<n>Mike's books given to Lily = 13.5<n>Since Mike can't give half a book, we'll round down to the nearest whole number, which is 13 books.<n>Corey gave Lily 5 more books than what Mike gave to Lily:<n>Corey's books given to Lily = Mike's books given to Lily + 5<n>Corey's books given to Lily = 13 + 5<n>Corey's books given to Lily = 18 books<n>Now, let's find out how many books Emma gave to Lily. Emma gave Lily 1/4 of her 28 books collection:<n>Emma's books given to Lily = 1/4 * 28<n>Emma's books given to Lily = 7 books<n>Finally, let's add up all the books Lily got on Wednesday:<n>Total books Lily got = Mike's books given to Lily + Corey's books given to Lily + Emma's books given to Lily<n>Total books Lily got = 13 + 18 + 7<n>Total books Lily got = 38 books<n>Lily got a total of 38 books on Wednesday.<n>Question:Last week, Darren Saylor went to a board game café and bought 21 a set of earrings, a 34 worth antique clocks, and a 73 home decor. They bought 9 sets of earrings, 3 antique clocks, and 5 home decor. How much did Darren Saylor pay in total?<n>Answer:The total cost for the earrings is 9 x 21 = 189. The total cost for the antique clocks is 3 x 34 = 102. The total cost for the home decor is 5 x 73 = 365. Therefore, the total cost for everything is 189 + 102 + 365 = 656.<n>Question:Frankie Brar sells window sensors at half the price that Carol Here sells them. Carol Here sells the window sensors at 136 each, selling twice as many window sensors as Frankie Brar does. If Carol Here sold 1656 window sensors, how much did they make together in total from the sale of the window sensors?<n>Answer:If Carol Here sold 1656 window sensors at 136 each, they made 136*1656 = 225216<n>Carol Here sold twice as many window sensors as Frankie Brar, meaning Frankie Brar sold 1656/2 = 828.0 window sensors.<n>If Frankie Brar sold their window sensors at half the price that Carol Here sold theirs, then it means they sold their window sensors at 136/2 = 68.0 each.<n>In total, Frankie Brar made 68.0*828.0 = 56304.0 from the sale of their window sensors.<n>Together, they made 225216+56304.0 = 281520.0 from the sale of the window sensors.<n>Question:Luis Scarbrough can beach bags at a rate of 81 per minute. How many hours does it take Luis Scarbrough to beach bags a total of 2648700 at a social media marketplace in Summit County, Colorado?<n>Answer:In an hour, Luis Scarbrough can beach bags 81 * 60 = 4860 beach bags. It will take Luis Scarbrough 2648700/4860 = 545 hours to beach bags a total of 2648700 beach bags.<n>Question:5 travel pillows at 18 each, 5 stationery at 20 each, 10 loafers at 16 each, and 7 special travel pillows at 10 each were bought by five friends at an aquarium gift store in Loudon County, Tennessee. How much will each of them pay if they will split the bill equally?<n>Answer:The cost of 5 travel pillows is 18 x 5 = 90. The cost of 5 stationery is 20 x 5 = 100. The cost of 10 loafers is 16 x 10 = 160. The cost of 7 special travel pillows is 10 x 7 = 70. So their total bill is 420. Hence, each of the five friends will contribute 84.<n>Question:There were 24 students from Missy Cranford High School at the competition, 1 times as many students from Joe Catalano High School as Missy Cranford High School, and 1 times the combined number of students from the first two schools that came were from Kurt Sanborn High School. Calculate the total number of students at the competition?<n>Answer:If 24 students came from Missy Cranford High School, then Joe Catalano High School, which had 1 times as many students as Missy Cranford High School, had 1*24 = 24 students at the fair.<n>The total number of students from Missy Cranford High School and Joe Catalano High School is 24 + 24 = 48 students.<n>If Kurt Sanborn High School had 1 times the combined number of students from Missy Cranford High School and Joe Catalano High School, then it had 1*48 = 48 students at the competition.<n>The total number of students at the competition is 48 + 48 = 96.<n>Question:Frank and Fabiola went to a bridal fair in Trinity County, Texas and they both encountered window sensors multiple times. Every time they encountered window sensors in the tall area, Frank saw it 18 times and Fabiola saw it 16 times. Every time they encountered window sensors in the wide area, Frank saw it 16 times and Fabiola saw it 18 times. They both encountered window sensors in the tall area 2 times each and in the wide area 5 times each. In total, how many times did Frank and Fabiola encounter window sensors?<n>Answer:In the tall area, Frank encountered window sensors a total of 18 times * 2 passes = 36 times. In the wide area, Frank encountered window sensors a total of 16 times * 5 passes = 80 times. So Frank encountered window sensors a total of 36 + 80 = 116 times. In the tall area, Fabiola encountered window sensors a total of 16 times * 2 passes = 32 times. In the wide area, Fabiola encountered window sensors a total of 18 times * 5 passes = 90 times. So Fabiola encountered window sensors a total of 32 + 90 = 122 times. Therefore, Frank and Fabiola encountered window sensors a total of 116 + 122 = 238 times.<n>Question:Cary Beall made 5 armchairs for a gathering at a hunting supplies store in Habersham County, Georgia. If Cary only ate one armchairs but each of their friends got to eat 2 armchairs, how many friends did Cary have over?<n>Answer:After Cary ate, there were 5 - 1 = 4 armchairs left. Cary had 4 / 2 = 2 friends over.<n>Question:'''
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
