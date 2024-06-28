
<div align="center">
<h1>
  Yuan2.0-M32: Mixture of Experts with Attention Router 
</h1>
</div>


<p align="center">
👾 <a href="https://www.modelscope.cn/profile/YuanLLM" target="_blank">ModelScope</a> • 🤗 <a href="https://huggingface.co/IEITYuan" target="_blank">Hugging Face</a> •  💬 <a href="https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/images/%E6%BA%90%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.png" target="_blank">WeChat</a>• 📎  <a href="https://arxiv.org/abs/2405.17976" target="_blank">Yuan2.0-M32 Paper</a>
</p>



<div align="center">

    
  <a href="code_license">
    <img alt="Code License" src="https://img.shields.io/badge/Apache%202.0%20-green?style=flat&label=Code%20License&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0-MoE%3Ftab%3DApache-2.0-1-ov-file"/>
  </a>
  <a href="model_license">
    <img alt="Model License" src="https://img.shields.io/badge/Yuan2.0%20License-blue?style=flat&logoColor=blue&label=Model%20License&color=blue&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0%2Fblob%2Fmain%2FLICENSE-Yuan" />
  </a>

</div>


<h4 align="center">
    <p>
        <b>English</b> |
        <a href="./README_CN.md">简体中文</a>
    <p>
</h4>


-----



##  0. Latest News 🎉🎉

* **[2024-06-18]** **Yuan2.0-M32-HF-INT4 released** 🎗️🎗️
* **[2024-05-28]** **Yuan2.0-M32 released**




##  1. Introduction


**Yuan2.0-M32** is a Mixture-of-Experts (MoE) language model with 32 experts, of which 2 are active. A new router network, Attention Router, is proposed and has been adopted for more efficient expert selection, boosting accuracy by 3.8% over models using a classical router network. Yuan 2.0-M32 is trained from scratch with 2000B tokens, and its training computation is only 9.25% of that required by a dense model of the same parameter scale. Demonstrating competitive capabilities in coding, math, and various specialized fields, Yuan2.0-M32 operates with only 3.7B active parameters out of a total 40B, and a forward computation of 7.4 GFLOPS per token, which is just 1/19th of Llama3-70B's requirement. Yuan 2.0-M32 has surpassed Llama3-70B on the MATH and ARC-Challenge benchmarks, achieving accuracies of 55.9% and 95.8%, respectively. The basic information of the **Yuan2.0-M32** model is as follows:

+ **Total Parameters ：** 40B <br>
+ **Experts：** 32 <br>
+ **Active Experts：** 2 <br>
+ **Active Parameters：** 3.7B <br>  
+ **Pretrained Tokens：** 2000B tokens <br>
+ **Sequence Length：** 16K <br>

The technical report for the Yuan2.0-M32 model has been released, and you can find more detailed technical information and evaluation results by referring to the <a href="https://arxiv.org/abs/2405.17976" target="_blank">**paper**</a>.



<div align=center> <img src=https://github.com/IEIT-Yuan/Yuan2.0-M32/blob/main/docs/Yuan2.0-M32-Architecture.jpg width=80% />

Fig.1: Yuan 2.0-M32 Architecture

</div>



##  2. Model Downloads


|    Model     | Sequence Length  |   Type   |         Download         |
| :----------: | :------: | :-------: |:---------------------------: |
| Yuan2.0-M32 |    16K    |    Megatron    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32/) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32) \| [Netdisk](https://pan.baidu.com/s/1K0LVU5NxeEujtYczF_T-Rg?pwd=cupw) \| [Wisemodel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32)
| Yuan2.0-M32-HF |    16K    | HuggingFace    |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-hf) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf) \| [Netdisk](https://pan.baidu.com/s/1FrbVKji7IrhpwABYSIsV-A?pwd=q6uh) \| [Wisemodel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-hf)
| Yuan2.0-M32-GGUF |    16K    | GGUF         |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-gguf/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-gguf) \| [Netdisk](https://pan.baidu.com/s/1BWQaz-jeZ1Fe69CqYtjS9A?pwd=f4qc) \| [Wisemodel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-gguf)
| Yuan2.0-M32-GGUF-INT4 |    16K    | GGUF    |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-gguf-int4/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-gguf-int4) \| [Netdisk](https://pan.baidu.com/s/1FM8xPpkhOrRcAfe7-zUgWQ?pwd=e6ag) \| [Wisemodel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-gguf-int4)
| Yuan2.0-M32-HF-INT4 |    16K    |  HuggingFace    |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-HF-INT4/summary)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf-int4) \| [Netdisk](https://pan.baidu.com/s/1zacOAxCne9U99LdgMbjfFQ?pwd=kkww )  \| [Wisemodel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-hf-int4/)
| Yuan2.0-M32-HF-INT8 |    16K    |  HuggingFace    |    [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-hf-int8/)  \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf-int8/) \| [Netdisk](https://pan.baidu.com/s/1hq9l6eYY_cRuBlQMRV6Lcg?pwd=b56k) \| [Wisemodel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-M32-hf-int8/)


\* __*Yuan2.0-M32-HF-INT4*__: The method of quantization and inference ，refer to the [guide](https://github.com/IEIT-Yuan/Yuan2.0-M32/blob/main/docs/README_GPTQ_EN.md).


##  3. Evaluation


**3.1 Benchmarks** 🏆


We conducted a thorough evaluation of the Yuan2.0-M32 model across a range of benchmarks, including HumanEval, GSM8K, MMLU, Math, and ARC-Challenge. These benchmarks are designed to test the model's proficiency in key areas such as natural language understanding, knowledge acquisition, mathematical computation and reasoning, and code generation. The Yuan2.0-M32 has shown a consistent and significant advantage over other models like Llama3-8B and Mistral-8×7B, excelling in all evaluated tasks. Remarkably, its overall performance is on par with the more substantial Llama3-70B model.The detailed evaluation results are outlined in the subsequent table.

- We provided evaluation scripts for [**HumanEval**](./docs/eval_humaneval.md), [**GSM8K**](./docs/eval_gsm8k.md), [**MMLU**](./docs/eval_mmlu.md), [**Math**](./docs/eval_math.md) and [**ARC-C**](./docs/eval_arc.md) to support the replication of our evaluation results.




| Model              |      HumanEval     |      GSM8K     |        MMLU       |         Math       |        ARC-C\*    |
| ------------------ |  :---------------: | :------------: | :---------------: |  :---------------: |  :---------------:|
| Llama3-70B         |     **81.7%**      |    **93%**     |       **80.3%**    |         50.4%      |         93.3%     |
| Llama3-8B          |        62.2%       |     79.6%      |       68.4%       |         30%        |         78.6%     |
| Phi-3-medium       |        62.2%       |     91.0%      |       78.0%       |         -          |         91.6%     |
| Phi-3-small        |        61%         |     89.6%      |       75.7%       |         -          |         90.7%     |
| Phi-3-mini         |        58.5%       |     82.5%      |       68.8%       |         -          |         84.9%     |
| Mistral-8*22B      |        45.1%       |     78.6%      |       77.8%       |         41.8%      |         91.3%     |
| Mistral-8*7B       |        40.2%       |     58.4%      |       70.86%      |         28.4%      |         85.9%     |
| **Yuan2.0-M32**    |        74.4%       |     92.7%      |       72.2%       |      **55.9%**     |       **95.8%**   |


\* __*ARC-C*__: Al2 Reasoning Challenge (ARC) benchmark is divided into Easy and Challenge parts, with the later containing more complex parts that needs further reasoning. We test our model on the Challenge parts.



-----

**3.2 Computational Utilization for Model** 

| Model              |      Params (B)    |  Active Params (B) | GFLOPs/token (Inference) | GFLOPS/token (Fine-tune) | Mean Accuracy	| Average Accuracy/GFLOPSs per token (Inference) |
| ------------------ |  :---------------: | :------------: | :---------------: |  :---------------: |  :---------------:|:---------------:|
| Llama3-70B         |         70         |     70         |       140      |       420      |      79.25       |       0.57     |
| Llama3-8B          |         8          |     8          |       16       |       48       |      64.15      |       4.00     |
| Mistral-8*22B      |         141        |     39         |       78       |       234      |      72.38      |       0.93     |
| Mistral-8*7B       |         47         |    12.9         |       25.8     |       77.3     |      60.83      |       2.36     |
| **Yuan2.0-M32**    |         40         |     3.7        |       7.4      |       22.2     |      79.15       |       10.69    |






##  4. Quick Start


**4.1  Environment Config**

We strongly recommend using the latest release of docker images of Yuan2.0-M32.You can launch an instance of the Yuan 2.0 container with the following Docker commands:

```bash
docker pull yuanmodel/yuan2.0:m32
docker run --gpus all --privileged --ulimit stack=68719476736 --shm-size=1000G -itd -v /path/to/yuan_2.0:/workspace/yuan_2.0 -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --name your_name yuanmodel/yuan2.0:m32
docker exec -it your_name bash
```


**4.2  Data Preprocess**

We have provided the data preprocess script. See documentation [here](./docs/data_process.md).

**4.3  Model Pretrain**

We've provided several scripts for pretraining in the [`example`](./examples). The details can be seen from documentation [here](./docs/pretrain.md).

**4.4  Inference Service**

- For a detailed deployment plan, please refer to [vllm](https://github.com/IEIT-Yuan/Yuan2.0-M32/blob/main/vllm/README_Yuan_vllm.md).
- Inference with Yuan2.0-M32-HF-INT4, please refer to [The Method of Quantization and Inference for Yuan2.0-M32](https://github.com/IEIT-Yuan/Yuan2.0-M32/blob/main/docs/README_GPTQ_EN.md).


##  5. Statement of Agreement


The use of the source code in this repository requires compliance with the open source license agreement Apache 2.0. The Yuan2.0 model supports commercial use and does not require authorization. Please understand and comply with the [《Yuan2.0 Model License Agreement》](./LICENSE-Yuan). Do not use the open source model and code, as well as derivatives generated from open source projects, for any purposes that may cause harm to the country and society, or for any services that have not undergone security assessment and filing. Although we have taken measures to ensure the compliance and accuracy of the data during training, the model has a huge number of parameters and is affected by probability and randomness factors. We cannot guarantee the accuracy of the output content, and the model is easily misled by input instructions. This project does not assume any data security, public opinion risks, or any model misleading, abusing, spreading caused by open-source models and code Risks and responsibilities arising from improper utilization You will be solely responsible for the risks and consequences arising from the use, copying, distribution, and modification of the model in this open source project



##  6. Contact Us 


**If you have any questions, please raise an issue or contact us at** air_service@ieisystem.com


##  7. Join Us 


We are currently recruiting experts in large model framework development, inference performance optimization, and open-source community operations. 

You can send resume to wushaohua@ieisystem.com, with the title of email: [Application of Yuan Team Application] - [Your Name].
