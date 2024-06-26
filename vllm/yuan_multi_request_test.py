from multiprocessing import Pool
from PIL import Image
import argparse
import os
from tqdm import tqdm
import time
import torch
import io
import numpy as np
import pickle
import re
from datetime import datetime
import time
import json
import requests

def test(prompt):
    raw_json_data = {
        "prompt": prompt,
        "logprobs": 1,
        "max_tokens": 500,
        "temperature": 1,
        "use_beam_search": False,
        "top_p": 0.00001,
        "top_k": 1,
        }
    json_data = json.dumps(raw_json_data)
    headers = {
    "Content-Type": "application/json",
    }
    response = requests.post(f'http://localhost:8000/generate', data=json_data, headers=headers)
    output = response.text
    return output

def main():
    bsz = 128
    prompts = ['请写一篇去武汉春游的活动规划']*bsz
    start_time = time.time()
    pool = Pool(processes=bsz)
    START_TIME = time.time()
    all_result = pool.map(test, prompts)
    pool.close()
    pool.join()
    end_time = time.time()
    print(all_result[0])
    print(all_result[-1])
    print(f"bsz={bsz}, inference_time: {end_time - start_time}, throughout: {bsz*500/(end_time - start_time)}")
    time.sleep(1)
    
if __name__ == '__main__':
    main()
