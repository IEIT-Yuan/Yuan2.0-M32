<h1 align="center">Yuan2-M32基于AutoGPTQ的量化和推理</h1>



## 配置AutoGPTQ环境
- AutoGPTQ环境配置要求：CUDA版本高于11.8
- 容器：使用[vllm](https://github.com/IEI-mjx/Yuan2.0-M32/blob/main/vllm/README_Yuan_vllm.md)项目提供的镜像创建容器
```shell
# 进入容器
docker exec -it vllm_yuan bash

# 进入你的工作目录
cd /mnt

# 拉取我们的项目
git clone https://github.com/IEIT-Yuan/Yuan2.0-M32.git

# 进入autogptq项目
cd  Yuan2.0-M32/3rd_party/AutoGPTQ

# 安装autogptq
pip install auto-gptq --no-build-isolation
```

## 量化Yuan2-M32-HF模型

量化Yuan2-M32模型主要分为三步：1.下载Yuan2-M32-HF模型 2.下载数据集 3.设置量化参数，量化Yuan2-M32-HF模型
- 1.下载Yuan2-M32 hugging face模型并移动到指定路径(/mnt/beegfs2/Yuan2-M32-HF)，可参考[vllm](https://github.com/IEI-mjx/Yuan2.0-M32/blob/main/vllm/README_Yuan_vllm.md)，模型下载地址：https://huggingface.co/IEIT-Yuan/Yuan2-M32-hf
- 2.数据集下载点击[这里](https://huggingface.co/datasets/hakurei/open-instruct-v1)，下载后移动到指定路径如：/mnt/beegfs2/
- 3.按照以下步骤调整量化参数进行量化操作
```shell
# 编辑Yuan2-M32-int4.py
cd /mnt/beegfs2/Yuan2.0-M32/3rd_party/AutoGPTQ
vim Yuan2-M32-int4.py

'''
pretrained_model_dir = "/mnt/beegfs2/Yuan2-M32-HF"
quantized_model_dir = "/mnt/beegfs2/Yuan2-M32-GPTQ-int4"

tokenizer = LlamaTokenizer.from_pretrained("/mnt/beegfs2/Yuan2-M32-HF", add_eos_token=False, add_bos_token=False, eos_token='<eod>', use_fast=True)
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

examples = []
with open("/mnt/beegfs2/instruct_data.json", 'r', encoding='utf-8') as file: # 数据集路径
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
# 1.修改pretrained_model_dir，指定量化后的quantized_model_dir
# 2.修改数据集路径
# 3.max_memory可以指定要使用的GPUs
# 4.修改量化参数，若要int4精度bits=4，若要int8精度bits=8，其他参数可以参考默认值

# 运行此脚本
python Yuan2-M32-int4.py

# 模型量化和packing过程耗时约8h，可以指定不同的GPU同时分别量化int4和int8
```


## GPTQ量化模型的推理
量化完成后，目标路径文件夹中会生成.safetensors后缀的ckpt文件以及config.json、quantize_config.json文件，需要先从Yuan2-M32-HF路径中拷贝tokenizer相关的文件
```shell
# 进入Yuan2-M32-HF路径
cd /mnt/beegfs2/Yuan2-M32-HF

# 拷贝tokenizer相关文件至Yuan2-M32-GPTQ-int4
cp special_tokens_map.json tokenizer* /mnt/beegfs2/Yuan2-M32-GPTQ-int4

# 编辑inference.py
cd /mnt/beegfs2/Yuan2.0-M32/3rd_party/AutoGPTQ
vim inference.py

'''
quantized_model_dir = "/mnt/beegfs2/Yuan2-M32-GPTQ-int4"

tokenizer = LlamaTokenizer.from_pretrained('/mnt/beegfs2/Yuan2-M32-GPTQ-int4', add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", trust_remote_code=True)
'''
# 修改quantized_model_dir和tokenizer路径

# 运行inference.py
python inference.py
```

## 推理精度测试
> HumanEval测试参数如下：
> generation_params = {
        "max_new_tokens": 512,
        "top_k": 1,
        "top_p": 0,
        "temperature": 1.0,
}

> BF16 双卡推理，GPTQ-INT4/INT8 单卡推理

> 测试结果：

| Model               | Accuracy Type |  HumanEval | Inference Speed |  Inference Memory Usage |
|---------------------|---------------|------------|-----------------|-------------------------|
| Yuan2-M32-HF        | BF16          |  73.17%    | 13.16 token/s   |76.34 GB                 |
| Yuan2-M32-GPTQ-int8 | INT8          |  72.56%    |  9.05 token/s   |39.81 GB                 |
| Yuan2-M32-GPTQ-int4 | INT4          |  66.46%    |  9.24 token/s   |23.27GB                  |



