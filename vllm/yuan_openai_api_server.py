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
        #outputs.append(output0)
        print(output)
        break
