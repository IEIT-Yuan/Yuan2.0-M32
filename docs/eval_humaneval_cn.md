# eval\_humaneval\_MOE中文

## 评测数据集

**datasets/HUMANEVAL/HumanEval.jsonl.gz** 英文原版的 [HumanEval](https://github.com/openai/human-eval "HumanEval") 评测数据集包含164 道问题。

**datasets/HUMANEVAL/HumanEval-textprompts.jsonl** 借助 gpt-4 翻译获得的中文版 HumanEval 数据集。

**datasets/HUMANEVAL/HumanEval-instructions.jsonl**  处理为指令跟随形式的HumanEval 数据集。

**datasets/HUMANEVAL/HumanEval-instructions.jsonl**  处理为指令跟随形式的HumanEval 数据集，并在提示词中加入了few-shot。

## 评测

### 简介

**examples/eval\_humaneval\_2xM32.sh.** 通过运行该程序，可以获得Yuan2.0-M32 模型在 HumanEval 评测数据集的评估结果。

在运行评测程序之前，你仅需在 bash 脚本中指定以下 checkpoint\_path参数，其他必要的路径已经设置好了：

| 参数名称              | 参数描述                 |
| ----------------- | -------------------- |
| `CHECKPOINT_PATH` | 待评测的checkpoint的保存路径。 |

### 环境要求

在 Yuan2.0-M32 checkpoint上运行 HumanEval 评测之前，确保已安装 HumanEval 程序。

```text
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
```

安装好HumanEval后，请移步至此脚本：

```纯文本
/usr/local/lib/python3.10/dist-packages/human_eval-1.0-py3.10.egg/human_eval/execution.py
```

并对 "check\_correctness "函数中的 "check\_program "变量作如下修改，以确保生成的代码中没有重复的函数签名。

```text
check_program = (
    #problem["prompt"] +
    completion + "\n" +
    problem["test"] + "\n" +
    f"check({problem['entry_point']})"
)

```

此外，如果您是第一次使用 HumanEval ，必须删除 "check\_program "中多余的 "#"，就在 "exec(check\_program, exec\_globals) "行之前。

```text
# WARNING
# This program exists to execute untrusted model-generated code. Although
# it is highly unlikely that model-generated code will do something overtly
# malicious in response to this test suite, model-generated code may act
# destructively due to a lack of model capability or alignment.
# Users are strongly encouraged to sandbox this evaluation suite so that it
# does not perform destructive actions on their host or network. For more
# information on how OpenAI sandboxes its code, see the accompanying paper.
# Once you have read this disclaimer and taken appropriate precautions,
# uncomment the following line and proceed at your own risk:
                         exec(check_program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

```



### 使用

运行以下命令评测 Yuan2.0-M32 模型在 HumanEval 数据集上的表现。运行 bash 脚本前，应将目录更改为 "Yuan2.0-M32 "主目录，并且需要在 bash 脚本中指定存放 checkpoint的路径。



在HumanEval数据集上评测Yuan2.0-M32模型：

```纯文本
cd <Specify Path>/Yuan2.0-M32/
bash examples/eval_humaneval_2x32.sh
```

### 结果

评测结果将收集在 \$OUTPUT\_PATH 中的 samples.jsonl 中。生成所有任务后，HumanEval 的 "evaluate\_functional\_correctness "函数将自动评测结果并返回准确度。

