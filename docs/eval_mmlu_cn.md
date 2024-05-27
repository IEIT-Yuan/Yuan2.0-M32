# eval_mmlu

## 数据集
**`datasets/MMLU/`.** 包含了 57 个子任务的英文评测数据集。

在每一个文本中，“[SEP]”之前的内容是原始问题，“[SEP]”之后的内容是该问题的标准答案。

## 评测

### 说明
**`examples/eval_mmlu_2x32B.sh`.** 运行该程序即可获得模型在mmlu数据集上的推理结果。

代码中的变量设置如下：

| 变量名            | 解释          |
| ------------------- | --------------------------------------------- |
| `CHECKPOINT_PATH`    | 待评测checkpoint的路径 |
| `TOKENIZER_MODEL_PATH` | tokenizer的路径          |
| `MATH_DATA`    | 待测试数据集的路径       |
| `OUTPUT_PATH`    | 推理结果的保存路径         |

### 运行

运行以下命令获得推理结果：
```
bash -x examples/eval_mmlu_2x32B.sh
```

### 结果
评测结果将保存在 `OUTPUT_PATH`中。其中，“[SEP]”之前的内容为原始问题，“[SEP]”之后的内容是模型对该问题的解析。

## 准确率
### 说明
**`tasks/MMLU/score_mmlu.py`.** 运行该程序即可获得mmlu评测结果的准确率。

代码中的变量设置如下：

| 变量名称               | 说明          |
| ------------------- | --------------------------------------------- |
| `origin_file_path`  | 测试集的保存路径               |
| `eval_file_path`    | 评测结果文件的保存路径       |
| `txt_eval_res_dir`  | 准确率评判结果的保存路径，以"true"结尾的文件中为正确结果，以"false"结尾的文件中为错误结果, csv和jsonl文件储存了每个子任务的准确率统计 |

### 运行
执行以下命令以评估模型在测试集上的准确率：
```
python score_mmlu.py
```
### 结果
“Number of correct answers”和“Number of incorrect answers”分别表示所有任务总体回答正确答案数和回答错误答案数，“accuracy”表示总体准确率。

