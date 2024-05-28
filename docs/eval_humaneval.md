# eval\_humaneval

## Dataset

**datasets/HUMANEVAL/HumanEval.jsonl.gz** The original English version of the [HumanEval](https://github.com/openai/human-eval "HumanEval") dataset containing 164 questions.

**datasets/HUMANEVAL/HumanEval-textprompts.jsonl** The Chinese version of the HumanEval dataset obtained by translation with the aid of gpt-4 model.

**datasets/HUMANEVAL/HumanEval-instructions.jsonl** The HumanEval dataset in instruction style.

**datasets/HUMANEVAL/HumanEval-instructions-fewshot.jsonl** The HumanEval dataset in instruction style with few-shot prompt.

## Evaluation

### Introduction

**examples/eval\_humaneval.sh.** The evaluation results for HumanEval on MOE model could be obtained by running this program.

Before running the evaluation program, you only have to specify the following checkpoint\_path in bash script by yourself, the other necessary paths are already set up.&#x20;

| Variable name     | Description                                         |
| ----------------- | --------------------------------------------------- |
| `CHECKPOINT_PATH` | the path that saves the checkpoint to be evaluated. |

### Requirement

Make sure HumanEval program is installed befere running the HumanEval evaluation on Yuan2.0 checkpoint.&#x20;

```text
$ git clone https://github.com/openai/human-eval
$ pip install -e human-eval
```

After HumanEval program is installed, we shall go to this script,

```纯文本
/usr/local/lib/python3.10/dist-packages/human_eval-1.0-py3.10.egg/human_eval/execution.py
```

and make the following change on "check\_program" variable in "check\_correctness" function, to ensure there is no duplicate function signature in generated codes.

```text
check_program = (
    #problem["prompt"] +
    completion + "\n" +
    problem["test"] + "\n" +
    f"check({problem['entry_point']})"
)

```

Also, if you are new to HumanEval, you have to delete the extra "#" in "check\_program", right before  the line "exec(check\_program, exec\_globals)".

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



### Usage

Run the following commands to evaluate the MOE model's performance on HumanEval dataset. Before running the bash script, you shall change directory to main directory of 'Yuan2.0-M32', and you have to specify the the checkpoit path where you store the checkpoint in the bash script.



Evaluate MOE model on HumanEval dataset.

```纯文本
cd <Specify Path>/Yuan2.0-M32/
bash examples/eval_humaneval_2x32.sh
```

### Results

The evaluation results will be gathered in samples.jsonl in \$OUTPUT\_PATH. After the generation of all the tasks done, the "evaluate\_functional\_correctness" function of HumanEval would automatically evaluate the results and return the accuracy.

