




### <font color=#FFC125 >源2.0-MoE 模型</font> 

-----
**ckpt转换说明**
### <strong>🔘 ckpt转换</strong> 

我们提供的的模型文件是8路流水并行（8pp）的模型文件，我们提供了自动转换脚本，可以依次执行完转换流程，使用方式如下：



**<font color=#FFFFF0 >如果提前将8路流水并行合并，可以直接执行： </font>**


```sh
bash examples/convert_hf_moe.sh
```

在转换时需要


**<font color=#FFFFF0 >如果不合并流水，可以按下面的方式进行转换： </font>**

首先执行转换脚本：

```sh
bash examples/convert_hf_moe.sh
```
执行这个脚本，每一路流水对应的.ckpt文件都会生成一个对应的.bin文件，等完成转换之后可以删除这些中间文件。

然后执行下面的命令：

```sh
python tools/concat.py --input-path $input_path --output-path $output_path --pp_rank 8 --num_layers 24 
```

这里的`--input-path`设置为上一步中产生的中间文件路径，这个命令会在`--output-path`设置的路径下生成一个完整的.bin文件。


### <strong>🔘 bin文件拆分</strong> 
执行上面的转换命令后会生成一个bin文件，可以执行下面的命令将其拆分：
```sh
python tools/split_bin.py --input-path $input_path --output-path $output_path
```

### <strong>🔘 HF模型推理</strong> 

可以通过如下代码调用YuanMoE模型来生成文本： 

```python
import torch, transformers
import sys, os
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from transformers import AutoModelForCausalLM,AutoTokenizer,LlamaTokenizer

print("Creat tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained('IEITYuan/Yuan2-hf-moe', add_eos_token=False, add_bos_token=False, eos_token='<eod>')
tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>','<commit_before>','<commit_msg>','<commit_after>','<jupyter_start>','<jupyter_text>','<jupyter_code>','<jupyter_output>','<empty_output>'], special_tokens=True)

print("Creat model...")
model = AutoModelForCausalLM.from_pretrained('IEITYuan/Yuan2-hf-moe', device_map='auto', torch_dtype=torch.bfloat16, trust_remote_code=True)

inputs = tokenizer("请问目前最先进的机器学习算法有哪些？", return_tensors="pt")["input_ids"].to("cuda:0")
outputs = model.generate(inputs,do_sample=False,max_length=100)
print(tokenizer.decode(outputs[0]))
```
