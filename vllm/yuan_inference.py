from vllm import LLM, SamplingParams
import time
import os
from transformers import LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('/mnt/beegfs2/Yuan2.0-M32-HF/', add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

prompts = ["写一篇春游作文"]
sampling_params = SamplingParams(max_tokens=300, temperature=1, top_p=0, top_k=1, min_p=0.0, length_penalty=1.0, repetition_penalty=1.0, stop="<eod>", )

llm = LLM(model="/mnt/beegfs2/Yuan2.0-M32-HF/", trust_remote_code=True, tensor_parallel_size=8, gpu_memory_utilization=0.8, disable_custom_all_reduce=True, max_num_seqs=1)

start_time = time.time()
outputs = llm.generate(prompts, sampling_params)
end_time = time.time()
total_tokens = 0
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    num_tokens = len(tokenizer.encode(generated_text, return_tensors="pt")[0])
    total_tokens += num_tokens
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("inference_time:", (end_time - start_time))
print("total_tokens:", total_tokens)
