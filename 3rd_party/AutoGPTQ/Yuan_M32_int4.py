from transformers import LlamaTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
import json
logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

pretrained_model_dir = "/mnt/beegfs2/Yuan2-M32-HF"
quantized_model_dir = "/mnt/beegfs2/Yuan2-M32-GPTQ-int4"

tokenizer = LlamaTokenizer.from_pretrained("/mnt/beegfs2/Yuan2-M32-HF", add_eos_token=False, add_bos_token=False, eos_token='<eod>', use_fast=True)
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

examples = []
with open("/mnt/beegfs2/instruct_data.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

for i, item in enumerate(data):
    if i >= 2000:
        break
    instruction = item.get('instruction', '')
    output = item.get('output', '')
    combined_text = instruction + " " + output
    examples.append(tokenizer(combined_text))

max_memory = {0: "80GIB", 1: "80GIB", 2: "80GIB", 3: "80GIB", 4: "80GIB", 5: "80GIB", 6: "80GIB", 7: "80GIB"}
quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
)

# load un-quantized model, by default, the model will always be loaded into CPU memory
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config, trust_remote_code=True, max_memory = max_memory)

# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples)

# save quantized model using safetensors
model.save_quantized(quantized_model_dir, use_safetensors=True)
# push quantized model to Hugging Face Hub.
# to use use_auth_token=True, Login first via huggingface-cli login.
# or pass explcit token with: use_auth_token="hf_xxxxxxx"
# (uncomment the following three lines to enable this feature)
# repo_id = f"YourUserName/{quantized_model_dir}"
# commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
# model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

# alternatively you can save and push at the same time
# (uncomment the following three lines to enable this feature)
# repo_id = f"YourUserName/{quantized_model_dir}"
# commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
# model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)

