from transformers import LlamaTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

quantized_model_dir = "/mnt/beegfs2/Yuan2-M32-GPTQ-int8"

tokenizer = LlamaTokenizer.from_pretrained('/tmnt/beegfs2/Yuan2-M32-GPTQ-int8', add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", trust_remote_code=True)

# inference with model.generate
print(tokenizer.decode(model.generate(**tokenizer("北京是中国的", return_tensors="pt").to(model.device), max_new_tokens=256)[0]))

# or you can also use pipeline
#pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
#print(pipeline("北京是中国的")[0]["generated_text"])

