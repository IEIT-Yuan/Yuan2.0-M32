


<div align="center">
<h1>
  源2.0-MoE 大模型
</h1>
</div>


<div align="center">

  <a href="https://airyuan.cn/home" target="_blank">
    <img alt="Homepage" src="https://img.shields.io/badge/Yuan2.0-red?style=flat&label=HomePage&link=https%3A%2F%2Fairyuan.cn%2Fhome" />
  </a>
  <a href="https://airyuan.cn/home" target="_blank">
    <img alt="Chat" src="https://img.shields.io/badge/Yuan2.0%20MoE-yellow?style=flat&logoColor=css&label=Chat&labelColor=7289da&link=https%3A%2F%2Fairyuan.cn%2Fhome" />
  <a href="code_license">
    <img alt="Code License" src="https://img.shields.io/badge/Apache%202.0%20-green?style=flat&label=Code%20License&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0-MoE%3Ftab%3DApache-2.0-1-ov-file"/>
  </a>
  <a href="model_license">
    <img alt="Model License" src="https://img.shields.io/badge/Yuan2.0%20License-blue?style=flat&logoColor=blue&label=Model%20License&color=blue&link=https%3A%2F%2Fgithub.com%2FIEIT-Yuan%2FYuan-2.0%2Fblob%2Fmain%2FLICENSE-Yuan" />
  </a>

</div>




<p align="center">
👾 <a href="https://www.modelscope.cn/profile/YuanLLM" target="_blank">ModelScope</a> • 🤗 <a href="https://huggingface.co/IEITYuan" target="_blank">Hugging Face</a> •  💬 <a href="resources/WECHAT.md" target="_blank">WeChat</a>• 📎  <a href="https://arxiv.org/ftp/arxiv/papers/2311/2311.15786.pdf" target="_blank">源2.0MoE论文</a>
</p>


<h4 align="center">
    <p>
        <b>简体中文</b> |
        <a href="./README-EN.md">English</a>
    <p>
</h4>





<!-- markdown-toc end -->
-----



##  0. Latest News 🎉🎉

* [2024-05-28] 发布源2.0-MoE 大模型





##  1. Introduction




浪潮信息 **“源2.0 M32”大模型（简称，Yuan2.0-M32）** 采用稀疏混合专家架构（MoE），以Yuan2.0-2B模型作为基底模型，通过创新的门控网络（YuanGate）实现32个专家间（Expers*32）的协同工作与任务调度，在显著降低模型推理算力需求的情况下，带来了更强的模型精度表现与推理性能；**★ 任务测评：** **Humaneval、GSM8K等业界领先**，  **★ 推理性能：** **2668 tokens/s，同规模模型性能最好**，**★ 推理成本：** **6.6 GFLOPs / token，相似精度模型成本最低**。**Yuan2.0-M32大模型** 基本信息如下：


+ **模型参数量：** 40B <br>
+ **专家数量：** 32 <br>
+ **激活专家数：** 2 <br>
+ **激活参数量：** 3.7B <br>  
+ **训练数据量：** 2481B tokens <br>
+ **微调数据量：** xxB tokens <br> 
+ **支持序列长度：** 256K <br>


同时，我们发布了Yuan2.0-M32模型的<a href="https://arxiv.org/ftp/arxiv/papers/2311/2311.15786.pdf" target="_blank">**技术报告**</a>，可以通过论文查看更详细的技术细节与测评结果。



Fig 1: Yuan 2.0-MoE 架构图


##  2. Model Downloads

**我们提供多种模型格式的下载链接：**

|    模型     | 序列长度  |   模型格式   |         下载链接         |
| :----------: | :------: | :-------: |:---------------------------: |
| Yuan2.0-M32 |    16K    |    Megatron    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-102B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-102B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-102B-hf)  \|  [百度网盘](https://pan.baidu.com/s/1O4GkPSTPu5nwHk4v9byt7A?pwd=pq74#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-102B-hf)  
| Yuan2.0-M32-HF |    16K    | HuggingFace    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-102B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-102B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-102B-hf)  \|  [百度网盘](https://pan.baidu.com/s/1O4GkPSTPu5nwHk4v9byt7A?pwd=pq74#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-102B-hf) 
| Yuan2.0-M32-GGUF |    16K    | GGUF    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-102B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-102B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-102B-hf)  \|  [百度网盘](https://pan.baidu.com/s/1O4GkPSTPu5nwHk4v9byt7A?pwd=pq74#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-102B-hf) 
| Yuan-2.0-MoE-GGUF-INT4 |    16K    | GGUF    | [ModelScope](https://modelscope.cn/models/YuanLLM/Yuan2.0-102B-hf/summary) \| [HuggingFace](https://huggingface.co/IEITYuan/Yuan2-102B-hf) \| [OpenXlab](https://openxlab.org.cn/models/detail/YuanLLM/Yuan2-102B-hf)  \|  [百度网盘](https://pan.baidu.com/s/1O4GkPSTPu5nwHk4v9byt7A?pwd=pq74#list/path=%2F) \| [WiseModel](https://www.wisemodel.cn/models/IEIT-Yuan/Yuan2-102B-hf) 



##  3. Evaluation Results


**3.1 Benchmarks 测试** 🏆


Yuan2.0-MoE模型与多个闭源、开源模型相比，均呈现出较好的精度表现。我们评测的数据集包括：Humaneval、GSM8K、MMLU、Math、ARC-Challenge，用于考察模型在自然语言理解、知识、数学计算和推理、代码生成等任务上的能力。Yuan2.0-MoE模型在所有测评任务上全面超越了Llama3-8B模型、Llama3-70B模型，并在xx项任务中达到了接近GPT-4的能力。



| Model              |      HumanEval     |      GSM8K     |        MMLU       |         Math       |    ARC-C\*  |
| ------------------ |  :---------------: | :------------: | :---------------: |  :---------------: |  :---------------:|
| GPT-4              |        86.2%       |     45.5%      |       15.2%       |         77.4%      |         -         |
| GPT3.5             |        92%         |     47.0%      |       16.1%       |         86.6%      |         59%       |
| Llama3-8B          |        68.6%       |     36.5%      |        7.3%       |         66.5%      |         34%       |
| Llama3-70B         |        56.8%       |       -        |         -         |         29.9%      |          -         |
| Mistral-7B         |        76.6%       |     38.7%      |       13.5%       |         67.1%      |         58%       |
| Mistral-8*7B       |        86.2%       |     45.5%      |       15.2%       |         77.4%      |         -         |
| Qwen1.5-72B        |        86.2%       |     45.5%      |       15.2%       |         77.4%      |         -         |
| Qwen1.5-MoE-A2.7B  |        86.2%       |     45.5%      |       15.2%       |         77.4%      |         -         |
| DeepSeek67B        |        86.2%       |     45.5%      |       15.2%       |         77.4%      |         -         |
| GLM-4              |        86.2%       |     45.5%      |       15.2%       |         77.4%      |         -         |
| **Yuan2.0-M32**    |     **86.2%**      |    **45.5%**   |     **15.2%**     |      **77.4%**    |       **15.2%**   |


\* __*ARC-C*__：ARC-Challenge， ARC数据集中的高阶测试问题，需要深层的推理能力和更广泛的知识背景。

-----

**3.2 长文本测试** 

基于旋转位置编码技术（RoPE），使用YaRN对原本的位置编码进行插值，使模型可以对超出预训练窗口长度的序列进行有效的位置编码,使用目标长度的文本数据对预训练模型进行微调，微调时使用插值后的旋转位置编码，并使用PoSE对训练数据进行处理。**Yuan2.0-M3 支持序列长度：** 256K 。

Fig2. “A Needle in the Haystack”

-----


**3.3 推理性能测试** 

相同条件下，对比 **Yuan2.0-M32** 、Mistral 8*7B、Llama3-8B、Llama2-70B 模型的推理性能，参数设置：__Batch size =1__，__Concurrency = 128__，测试结果如下：


|     类型    |       MoE 模型      |     MoE 模型    |    Dense模型     |     Dense模型    |
| :-------:  |:------------------:|:---------------:|:---------------:|:----------------:|
| 模型        |  **Yuan2.0-M32**  |   Mistral 8*7B  |    Llama3-8B    |    Llama3-70B  |
| 参数量      |      408亿         |       467亿     |     80.3亿      |      706亿     |
| 激活参数量   |    37亿            |      ~130亿      |     80.3亿      |     706亿      |
| 推理性能    |  **2668 tokens/s** | 1659 tokens/s | 2105 tokens/s | 1423 tokens/s  |


-----



**3.4 计算成本测试** 


Yuan2.0-M32中采用Yuan2.0-2B模型作为Expert，推理时激活2个Expert，激活参数量为3.7B，推理时所使用的激活参数量决定了推理成本，对比其他MoE与Dense模型，计算资源消耗最小，实测数据如下：


|     类型    |       MoE 模型      |     MoE 模型    |    Dense模型     |     Dense模型    |
| :-------:  |:------------------:|:---------------:|:---------------:|:----------------:|
| 模型        |   **Yuan2.0-M32**  |   Mistral 8*7B  |    Llama3-8B    |    Llama3-70B  |
| 参数量      |      408亿          |       467亿     |     80.3亿      |      706亿     |
| 计算资源    | **6.6 GFLOPs/token** | 21.7 GFLOPs/token | 13.9 GFLOPs/token | 120.2 GFLOPs/token  |

-----




##  4. Quick Start


**6.1  环境配置**



**6.2  数据预处理**



**6.3  模型预训练**



**6.4  推理服务**



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
