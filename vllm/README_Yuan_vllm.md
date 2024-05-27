# 基于vLLM的Yuan2-M32-HF推理服务部署

## 配置vLLM环境
vLLM环境配置主要分为以下两步，拉取我们提供的镜像创建docker，以及安装vllm运行环境
### Step 1. 拉取我们提供的镜像并创建容器

```bash
# 拉取我们提供的镜像
docker pull yuanmodel/vllm-v0.4.0:latest

# 创建容器, 可以用"-v"挂载至你的本地目录（此处挂载了mnt路径）
docker run --gpus all -itd --network=host  -v /mnt:/mnt --cap-add=IPC_LOCK --device=/devinfiniband --privileged 
--name vllm_yuan --ulimit core=0 --ulimit memlock=1 --ulimit stack=68719476736 --shm-size=1000G yuanmodel/vllm-v0.4.0:latest

# 进入容器
docker exec -it vllm_yuan bash
```
### Step 2. 安装vLLM运行环境

```bash
# 进入你的工作目录
cd /mnt

# 拉取我们的项目
git clone https://github.com/IEIT-Yuan/Yuan2.0-M32.git

# 进入vLLM项目
cd Yuan2.0-M32/vllm

# 安装依赖
CUDA_HOME=/usr/local/cuda MAX_JOBS=64 NVCC_THREADS=8 python setup.py install

# 拷贝.so文件
cp build/lib.linux-x86_64-3.10/vllm/*.so vllm/
```

## Yuan2-M32-HF模型基于vLLM的推理和部署

以下是如何使用vLLM推理框架对Yuan2.0-2Bx32模型进行推理和部署的示例

### Step 1. 准备Yuan2-M32的hf模型
下载Yuan2-M32-HF hugging face模型，参考地址：https://huggingface.co/IEIT-Yuan/Yuan2-M32-hf

将下载好的Yuan2-M32-HF模型的ckpt移动至你的本地目录下(本案例中的路径如下：/mnt/beegfs2/)
### Step 2. 基于Yuan2-M32-HF的vllm推理
#### Option1:单个prompt推理
```bash
# 编辑test_yuan_1prompt.py
vim yuan_inference.py

# 1.修改LLM模型路径(必选)
# 2.修改提示词prompts的内容
# 3.修改sampling_params参数(可选),其中max_num_seqs为vllm同时处理的序列数量
'''
prompts = ["写一篇春游作文"]
sampling_params = SamplingParams(max_tokens=300, temperature=1, top_p=0, top_k=1, min_p=0.0, length_penalty=1.0, repetition_penalty=1.0, stop="<eod>", )

llm = LLM(model="/mnt/beegfs2/Yuan2-M32-HF/", trust_remote_code=True, tensor_parallel_size=8, gpu_memory_utilization=0.8, disable_custom_all_reduce=True, max_num_seqs=64)
'''
# 注意：如用多个prompt进行推理时，可能由于补padding的操作，和用单个prompt推理时结果不一样
```
参数解释：

-tensor_parallel_size:张量并行大小

-gpu_memory_utilization:gpu的内存利用率，显存不足时可以调低此值

-max_num_seqs:最大同时处理的序列数量，在处理多prompts时这个参数决定了一次性可以处理的prompt个数，设置过高可能会导致OOM

-disable-custom-all-reduce:禁用自定义的all-reduce内核，并回退到NCCL


### Step 3. 基于vllm.entrypoints.api_server部署Yuan2-M32-HF
基于api_server部署Yuan2-M32-HF的步骤包括推理服务的发起和调用。其中调用vllm.entrypoints.api_server推理服务有以下两种方式：第一种是通过命令行直接调用；第二种方式是通过运行脚本批量调用。
```bash
# 发起服务，--model修改为您的ckpt路径
python -m vllm.entrypoints.api_server --model=/mnt/beegfs2/Yuan2-M32-HF/ --trust-remote-code --disable-custom-all-reduce --tensor-parallel-size=8 --max-num-seqs=1 --gpu-memory-utilization=0.8

# 发起服务后，服务端显示如下：
INFO 05-16 19:55:04 ray_gpu_executor.py:228] # GPU blocks: 8073, # CPU blocks: 682
INFO 05-16 19:55:04 model_runner.py:970] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 05-16 19:55:04 model_runner.py:974] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(RayWorkerVllm pid=13288) INFO 05-16 19:55:04 model_runner.py:970] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(RayWorkerVllm pid=13288) INFO 05-16 19:55:04 model_runner.py:974] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 05-16 19:55:05 model_runner.py:1061] Graph capturing finished in 0 secs.
(RayWorkerVllm pid=13288) INFO 05-16 19:55:05 model_runner.py:1061] Graph capturing finished in 0 secs.
INFO:     Started server process [709]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```
#### Option 1. 基于命令行调用服务
```bash
# 使用curl命令行调用服务的指令如下：
curl http://localhost:8000/generate -d '{"prompt": "写一篇春游作文", "use_beam_search": false,  "n": 1, "temperature": 1, "top_p": 0, "top_k": 1,  "max_tokens":256, "stop": "<eod>"}'

# 服务发起端输出结果显示如下：
INFO 05-17 09:20:11 async_llm_engine.py:508] Received request 94247f30ded54154ae2631cd65903a35: prompt: '写一篇春游作文', sampling_params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1, top_p=0, top_k=1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['<eod>'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=256, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: None, lora_request: None.
INFO 05-17 09:20:14 async_llm_engine.py:120] Finished request 94247f30ded54154ae2631cd65903a35.
INFO:     127.0.0.1:37938 - "POST /generate HTTP/1.1" 200 OK
```
#### Option 2. 基于命令脚本调用服务
调用api_server的相关脚本为yuan_api_server.py，内容如下
```bash
import requests
import json

outputs = []
with open('/mnt/Yuan2.0-M32/vllm/humaneval/human-eval-gpt4-translation-fixed5.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        prompt = data.get('prompt')
        raw_json_data = {
                "prompt": prompt,
                "logprobs": 1,
                "max_tokens": 256,
                "temperature": 1,
                "use_beam_search": False,
                "top_p": 0,
                "top_k": 1,
                "stop": "<eod>",
                }
        json_data = json.dumps(raw_json_data)
        headers = {
                "Content-Type": "application/json",
                }
        response = requests.post(f'http://localhost:8000/generate',
                             data=json_data,
                             headers=headers)
        output = response.text
        output = json.loads(output)
        output = output['text']
        outputs.append(output[0])
        break # 跳出循环，只读一条数据
print(outputs)   # 可以选择打印输出还是储存到新的jsonl文件
...

# 示例中是读取的中文版本humaneval测试集，通过批量调用推理服务并将结果保存在对应的jsonl文件中
# 您可以将代码中读取的jsonl文件路径替换为您的路径
# 或者在此代码基础上进行修改，例如手动传入多个prompts以批量调用api_server进行推理
```
修改完成后运行以下命令脚本调用推理服务即可
```bash
# 运行脚本，调用推理服务
python yuan_api_server.py

# 服务端调用输出显示
INFO 05-17 09:23:58 async_llm_engine.py:508] Received request 52d849c4be704318af0181eab1d414b8: prompt: '问题描述：检查给定数字列表中，是否有任何两个数字之间的距离小于给定的阈值。\n示例：\n>>> has_close_elements([1.0, 2.0, 3.0], 0.5)\nFalse\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue\n完成上述任务要求的Python代码。\n代码如下：\n```python\nfrom typing import List\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n', sampling_params: SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1, top_p=0, top_k=1, min_p=0.0, seed=None, use_beam_search=False, length_penalty=1.0, early_stopping=False, stop=['<eod>'], stop_token_ids=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=256, min_tokens=0, logprobs=1, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None), prompt_token_ids: None, lora_request: None.
INFO 05-17 09:24:02 async_llm_engine.py:120] Finished request 52d849c4be704318af0181eab1d414b8.
INFO:     ::1:39948 - "POST /generate HTTP/1.1" 200 OK
```
### Step 4. 基于vllm.entrypoints.openai.api_server部署Yuan2-M32-HF
注意：openai.api_server调用tokenizer方式有所不同，在使用此服务时，先将tokenizer_config.json中的add_eos_token和add_bos_token设置为false
```bash
# 修改tokenizer_config.json文件
vim /mnt/beegfs2/Yuan2-M32-HF/tokenizer_config.json
# 以下内容修改为false即可
"add_bos_token": false,
"add_eos_token": false,
```
基于openai的api_server部署Yuan2.0-2B的步骤和step 3的步骤类似，发起服务和调用服务的方式如下：

发起服务命令：
```bash
# 发起服务，--model修改为您的ckpt路径
python -m vllm.entrypoints.openai.api_server --model=/mnt/beegfs2/Yuan2-M32-HF/ --trust-remote-code --disable-custom-all-reduce --tensor-parallel-size=8 --max-num-seqs=1 --gpu-memory-utilization=0.8

# 发起服务后，服务端显示如下：
INFO 05-17 09:38:43 ray_gpu_executor.py:228] # GPU blocks: 8073, # CPU blocks: 682
INFO 05-17 09:38:44 model_runner.py:970] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 05-17 09:38:44 model_runner.py:974] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(RayWorkerVllm pid=42632) INFO 05-17 09:38:44 model_runner.py:970] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
(RayWorkerVllm pid=42632) INFO 05-17 09:38:44 model_runner.py:974] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
(RayWorkerVllm pid=43401) INFO 05-17 09:38:30 model_runner.py:169] Loading model weights took 9.7006 GB [repeated 6x across cluster]
(RayWorkerVllm pid=42632) INFO 05-17 09:38:44 model_runner.py:1061] Graph capturing finished in 0 secs.
INFO 05-17 09:38:44 model_runner.py:1061] Graph capturing finished in 0 secs.
WARNING 05-17 09:38:48 serving_chat.py:340] No chat template provided. Chat API will not work.
INFO:     Started server process [30051]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)

```
调用服务命令：
```bash
# 使用curl命令行调用服务的指令如下
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{"model": "/mnt/beegfs2/Yuan2-M32-HF/", "prompt": "写一篇春游作文", "max_tokens": 300, "temperature": 1, "top_p": 0, "top_k": 1, "stop": "<eod>"}'
```
调用服务脚本如下：
```bash
import requests
import json

outputs = []
with open('/mnt/Yuan2.0-M32/vllm/humaneval/human-eval-gpt4-translation-fixed5.jsonl', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        prompt = data.get('prompt')
        raw_json_data = {
                "model": "/mnt/beegfs2/Yuan2-M32-HF/",
                "prompt": prompt,
                "max_tokens": 256,
                "temperature": 1,
                "use_beam_search": False,
                "top_p": 0,
                "top_k": 1,
                "stop": "<eod>",
                }
        json_data = json.dumps(raw_json_data, ensure_ascii=True)
        headers = {
                "Content-Type": "application/json",
                }
        response = requests.post(f'http://localhost:8000/v1/completions',
                             data=json_data,
                             headers=headers)
        output = response.text
        output = json.loads(output)
        output = output["choices"][0]['text']
        print(output)
        break
...
# 此脚本您需要修改"model"后的ckpt路径，其他修改方式和yuan_api_server.py一致
```




