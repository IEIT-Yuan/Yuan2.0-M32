
<div align="center">
<h1>
  源2.0 M32大模型
</h1>
</div>


<p align="center">
👾 <a href="https://www.modelscope.cn/profile/YuanLLM" target="_blank">ModelScope</a> • 🤗 <a href="https://huggingface.co/IEITYuan" target="_blank">Hugging Face</a> •  💬 <a href="https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/images/%E6%BA%90%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.png" target="_blank">WeChat</a>• 📎  <a href="https://github.com/IEIT-Yuan/Yuan2.0-M32/blob/main/docs/Paper.pdf" target="_blank">源2.0 M32论文</a>
</p>



<div align="center">

    
  <a href="code_license">
    <img alt="Code License" src="https://img.shields.io/badge/Apache%202.0%20-green?style=flat&label=Code%20License&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0-MoE%3Ftab%3DApache-2.0-1-ov-file"/>
  </a>
  <a href="model_license">
    <img alt="Model License" src="https://img.shields.io/badge/Yuan2.0%20License-blue?style=flat&logoColor=blue&label=Model%20License&color=blue&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0%2Fblob%2Fmain%2FLICENSE-Yuan" />
  </a>

</div>









-----



##  0. Latest News 🎉🎉

* [2024-05-28] 发布源2.0 M32 大模型




##  1. Introduction


浪潮信息 **“源2.0 M32”大模型（简称，Yuan2.0-M32）** 采用稀疏混合专家架构（MoE），以Yuan2.0-2B模型作为基底模型，通过创新的门控网络（Attention Router）实现32个专家间（Expers*32）的协同工作与任务调度，在显著降低模型推理算力需求的情况下，带来了更强的模型精度表现与推理性能；源2.0-M32在多个业界主流的评测进行了代码生成、数学问题求解、科学问答与综合知识能力等方面的能力测评。结果显示，源2.0-M32在多项任务评测中，展示出了较为先进的能力表现，MATH（数学求解）、ARC-C（科学问答）测试成绩超越LLaMA3-700亿模型。**Yuan2.0-M32大模型** 基本信息如下：

+ **模型参数量：** 40B <br>
+ **专家数量：** 32 <br>
+ **激活专家数：** 2 <br>
+ **激活参数量：** 3.7B <br>  
+ **训练数据量：** 2000B tokens <br>
+ **支持序列长度：** 16K <br>


同时，我们发布了Yuan2.0-M32模型的<a href="https://github.com/IEIT-Yuan/Yuan2.0-M32/blob/main/docs/Paper.pdf" target="_blank">**技术报告**</a>，可以通过论文查看更详细的技术细节与测评结果。


<div align=center> <img src=https://github.com/IEIT-Yuan/Yuan2.0-M32/blob/main/docs/Yuan2.0-M32-Architecture.jpg width=80% />

Fig.1: Yuan 2.0-M32 架构图

</div>



##  2. Model Downloads

**我们提供多种模型格式的下载链接：**

|    模型     | 序列长度  |   模型格式   |         下载链接         |
| :----------: | :------: | :-------: |:---------------------------: |
| Yuan2.0-M32 |    16K    |    Megatron    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32/) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32) \| [百度网盘](https://pan.baidu.com/s/1K0LVU5NxeEujtYczF_T-Rg?pwd=cupw)
| Yuan2.0-M32-HF |    16K    | HuggingFace    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2-M32-hf) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-M32-hf) \| [百度网盘](https://pan.baidu.com/s/1FrbVKji7IrhpwABYSIsV-A?pwd=q6uh)
| Yuan2.0-M32-GGUF |    16K    | GGUF     |    [百度网盘](https://pan.baidu.com/s/1BWQaz-jeZ1Fe69CqYtjS9A?pwd=f4qc)
| Yuan2.0-M32-GGUF-INT4 |    16K    | GGUF  |   [百度网盘](https://pan.baidu.com/s/1FM8xPpkhOrRcAfe7-zUgWQ?pwd=e6ag)




##  3. Evaluation Results


**3.1 Benchmarks 测试** 🏆


Yuan2.0-M32 模型与多个闭源、开源模型相比，均呈现出较好的精度表现。我们评测的数据集包括：Humaneval、GSM8K、MMLU、Math、ARC-Challenge，用于考察模型在自然语言理解、知识、数学计算和推理、代码生成等任务上的能力。Yuan2.0-M32模型在所有测评任务上全面超越了Llama3-8B、Mistral-8*7B等模型，综合能力表现可以对标 Llama3-70B模型。



| Model              |      HumanEval     |      GSM8K     |        MMLU       |         Math       |        ARC-C\*    |
| ------------------ |  :---------------: | :------------: | :---------------: |  :---------------: |  :---------------:|
| Llama3-70B         |     **81.7%**      |    **93%**     |       **80.3**    |         50.4%      |         93.3%     |
| Llama3-8B          |        62.2%       |     79.6%      |       68.4%       |         30%        |         78.6%     |
| Phi-3-medium       |        62.2%       |     91.0%      |       78.0%       |         -          |         91.6%     |
| Phi-3-small        |        61%         |     89.6%      |       75.7%       |         -          |         90.7%     |
| Phi-3-mini         |        58.5%       |     82.5%      |       68.8%       |         -          |         84.9%     |
| Mistral-8*22B      |        45.1%       |     78.6%      |       77.8%       |         41,8%      |         91.3%     |
| Mistral-8*7B       |        40.2%       |     58.4%      |       70.86%      |         28.4%      |         85.9%     |
| **Yuan2.0-M32**    |        74.4%       |     92.7%      |       72.2%       |      **55.9%**     |       **95.8%**   |


\* __*ARC-C*__：ARC-Challenge， ARC数据集中的高阶测试问题，需要深层的推理能力和更广泛的知识背景。

-----

**3.2 模型算力效率** 

| Model              |      Params (B)    |  Active Params (B) | GFLOPs/token (Inference) | GFLOPs/token (Fine-tune) | Mean Accuracy	| Mean Accuracy  GFLOPs per token (Inference) |
| ------------------ |  :---------------: | :------------: | :---------------: |  :---------------: |  :---------------:|:---------------:|
|                    |         参数量      |     激活参数量   | 算力消耗/token （推理阶段） | 算力消耗/token （微调阶段） |    平均测评分数     |	  模型算力效率     |
| Llama3-70B         |         70         |     70         |       140      |       420      |      79.25       |       0.57     |
| Llama3-8B          |         8          |     8          |       16       |       48       |      64.15      |       4.00     |
| Mistral-8*22B      |         141        |     39         |       78       |       234      |      72.38      |       0.93     |
| Mistral-8*7B       |         47         |    12.9         |       25.8     |       77.3     |      60.83      |       2.36     |
| **Yuan2.0-M32**    |         40         |     3.7        |       7.4      |       22.2     |      79.15       |       10.69    |






##  4. Quick Start


**4.1  环境配置**

我们建议使用yuan2.0-M32的最新docker.

我们可以通过下面命令启动容器：

```bash
docker pull yuanmodel/yuan2.0:m32
docker run --gpus all --privileged --ulimit stack=68719476736 --shm-size=1000G -itd -v /path/to/yuan_2.0:/workspace/yuan_2.0 -v /path/to/dataset:/workspace/dataset -v /path/to/checkpoints:/workspace/checkpoints --name your_name yuanmodel/yuan2.0:m32
docker exec -it your_name bash
```


**4.2  数据预处理**

我们提供了数据预处理的脚本，参考[数据预处理说明文档](./docs/data_process.md).

**4.3  模型预训练**

我们提供了用于预训练的文档和 [`example`](./examples)的脚本，具体使用方法可以参考[预训练说明文档](./docs/pretrain.md).

**4.4  推理服务**

-详细部署方案可以参考[vllm](https://github.com/IEIT-Yuan/Yuan2.0-M32/edit/main/vllm/README_Yuan_vllm.md)


##  5. Statement of Agreement

使用源2.0代码及模型需遵循 [Apache 2.0](https://github.com/xxxxxxE) 开源协议和[《源2.0模型许可协议》](./LICENSE-Yuan)，源2.0模型支持商用，不需要申请授权，请您了解并遵循，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。

尽管模型在训练时我们已采取措施尽力确保数据的合规性和准确性，但模型参数量巨大且受概率随机性因素影响，我们无法保证输出内容的准确性，且模型易被输入指令所误导，本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。**您将对通过使用、复制、分发和修改模型等方式利用该开源项目所产生的风险与后果，独自承担全部责任。**



##  6. Developer Program

**源大模型共训计划** 🚀

我们希望开源的模型更符合开发者应用需求，为此我们推出源大模型共训计划，开发者提出自己的应用或场景的需求，由我们来准备训练数据并对源大模型进行增强训练，训练后的模型依然在社区开源。

每月六日我们会收集前一月开发者提出的具体需求，经过评审后列入当月模型训练计划，训练完成后的模型在当月月末就会更新到开源社区。开发者只需要提出需求，由我们来进行数据准备、模型训练并开源。请开发者在issue的“源大模型共训计划”问题下提出具体需求，提出需求的具体格式无要求，只需要说清楚具体的应用场景、对大模型的能力需求以及给出输入输出的说明。

🕙 **以下是提出需求的一些示例：**

**1. 场景需求**：能够基于业务场景生成相关内容，对场景的描述。
 输入：用户问题，输出：正确的答案。

**2. 场景需求**：我想让大模型能够阅读一个领域下的多篇论文，给出这些论文的综述，当前领域研究的热点以及未解决的问题，从而辅助学术研究。
输入为：一个领域下的多篇论文，输出为：综述研究报告，研究热点总结，
未解决问题总结。

**3. 场景需求**：...... 能够反应场景的典型特性即可



##  7. Contact Us 


**1. 给我们发邮件：** air_service@ieisystem.com

**2.加入开发者微信群：**
扫码关注“源AI看世界”公众号，发送消息 **“入群”** 获取开发者技术交流群二维码。</br>
  ![Image text](https://github.com/IEIT-Yuan/Yuan-2.0/blob/main/images/%E6%BA%90%E5%85%AC%E4%BC%97%E5%8F%B7%E4%BA%8C%E7%BB%B4%E7%A0%81.png)



##  8. Join Us 

我们正在招聘大模型框架研发、推理性能优化、开源社区运营方向相关专家。

请申请者将个人简历发送至邮箱(wushaohua@ieisystem.com)，并注明邮件主题”源项目团队应聘简历-个人名字”。
