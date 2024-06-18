<h1 align="center">The Method of Quantization and Inference for Yuan2.0-M32</h1>


<div align="center">

    
  <a href="code_license">
    <img alt="Code License" src="https://img.shields.io/badge/Apache%202.0%20-green?style=flat&label=Code%20License&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0-MoE%3Ftab%3DApache-2.0-1-ov-file"/>
  </a>
  <a href="model_license">
    <img alt="Model License" src="https://img.shields.io/badge/Yuan2.0%20License-blue?style=flat&logoColor=blue&label=Model%20License&color=blue&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0%2Fblob%2Fmain%2FLICENSE-Yuan" />
  </a>

</div>



##  0. Model Downloads


|    Model     | Sequence Length  |   Type   |         Download         |
| :----------: | :------: | :-------: |:---------------------------: |
| Yuan2.0-M32-HF-INT4 |    16K    | HuggingFace    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-HF-INT4/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf-int4) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-hf-int4/) 


## 1. Environment of AutoGPTQ 
- **Environment requirements:**  CUDA version> 11.8 
- **Container:**  Create a container using the image provided by the [vllm](https://github.com/IEI-mjx/Yuan2.0-M32/blob/main/vllm/README_Yuan_vllm.md)
```shell
# enter docker containers
docker exec -it vllm_yuan bash

# enter directory
cd /mnt

# clone
git clone https://github.com/IEIT-Yuan/Yuan2.0-M32.git

# enter project
cd  Yuan2.0-M32/autogptq

# install autogptq
pip install auto-gptq --no-build-isolation
```

## 2. Quantize Yuan2.0-M32-HF model





**The Steps for Quantizing Yuan2.0-M32 Model:**

- **Step 1:** Download [Yuan2.0-M32-HF](https://github.com/IEIT-Yuan/Yuan2.0-M32?tab=readme-ov-file#2-model-downloads) Model and move it to the specified path (/mnt/beegfs2/Yuan2-M32-HF), refer to [vllm](https://github.com/IEI-mjx/Yuan2.0-M32/blob/main/vllm/README_Yuan_vllm.md)
- **Step 2:** Download the [datases](https://huggingface.co/datasets/hakurei/open-instruct-v1), then move it to the specified path (/mnt/beegfs2/)
- **Step 3:** Adjust the parameters according to the following script for the quantization operation.



```shell
# edit Yuan2-M32-int4.py
cd /mnt/beegfs2/Yuan2.0-M32/autogptq
vim Yuan2-M32-int4.py

'''
pretrained_model_dir = "/mnt/beegfs2/Yuan2-M32-HF"
quantized_model_dir = "/mnt/beegfs2/Yuan2-M32-GPTQ-int4"

tokenizer = LlamaTokenizer.from_pretrained("/mnt/beegfs2/Yuan2-M32-HF", add_eos_token=False, add_bos_token=False, eos_token='<eod>', use_fast=True)

examples = []
with open("/mnt/beegfs2/instruct_data.json", 'r', encoding='utf-8') as file: # path of datasets
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
'''

# Modify pretrained_model_dir, specify the quantized_model_dir for the quantized model.
# Modify the path of the datasets.
# max_memory can specify the GPUs to be used.
# Adjust the quantization parameters, int4: set bits=4, int8: set bits=8. 
# Other parameters can refer to the default values.


# Run
python Yuan2-M32-int4.py

# The model quantization and packing process takes 8 hours approximately.
# You can use GPUs quantize the model to int4 and int8 separately at the same time.
```


## 3. Inference with Quantized Model

Quantization completed, you will get the checkpoint files with the suffix of '.safetensors', config.json and quantize_config.json in the folder. You need to first copy the tokenizer-related files from the Yuan2-M32-HF path.


```shell
# the path of Yuan2-M32-HF
cd /mnt/beegfs2/Yuan2-M32-HF

# copy tokenizer files to the path of Yuan2-M32-GPTQ-int4
cp special_tokens_map.json tokenizer* /mnt/beegfs2/Yuan2-M32-GPTQ-int4

# edit inference.py
cd /mnt/beegfs2/Yuan2.0-M32/autogptq
vim inference.py

'''
quantized_model_dir = "/mnt/beegfs2/Yuan2-M32-GPTQ-int4"

tokenizer = LlamaTokenizer.from_pretrained('/mnt/beegfs2/Yuan2-M32-GPTQ-int4', add_eos_token=False, add_bos_token=False, eos_token='<eod>')

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", trust_remote_code=True)
'''
# edit paths of quantized_model_dir and tokenizer

# run inference.py
python inference.py
```

## 4. Evaluation
> Parameters of HumanEval:
> generation_params = {
        "max_new_tokens": 512,
        "top_k": 1,
        "top_p": 0,
        "temperature": 1.0,
}

> Result:

| Model               | Accuracy Type | Ckpt Size | HumanEval |
|---------------------|---------------|-----------|-----------|
| Yuan2-M32-HF        | BF16          | 75GB      | 72.56%    |
| Yuan2-M32-GPTQ-int8 | INT8          | 40GB      | 72.56%    |
| Yuan2-M32-GPTQ-int4 | INT4          | 21GB      | 66.46%    |


