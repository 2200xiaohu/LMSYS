import argparse
from typing import Optional, Union

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass

import datasets
from datasets import Dataset
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import log_loss

from transformers import (
    AutoTokenizer,
    AutoConfig,
    EarlyStoppingCallback,
    AutoModelForCausalLM,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    RobertaForMultipleChoice,
    AutoModelForSequenceClassification,
    LlamaModel,
    LlamaForSequenceClassification,
    BitsAndBytesConfig,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    TrainerCallback,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy

from peft import (
    get_peft_config,
    PeftModel,
    PeftConfig,
    get_peft_model,
    LoraConfig,
    TaskType,
)
import os

import random
from random import randint
def seed_everything(seed=None):
    '''
    固定seed
    :param seed: int, 随机种子
    '''
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if (seed is None) or not (min_seed_value <= seed <= max_seed_value):
        seed = random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


seed_everything(42)

from utils import load_split_data, load_json

from torch.utils.data import Dataset
class InstructionDataSet(Dataset):
    def __init__(self, data, tokenizer, max_source_length, max_target_length):
        super(InstructionDataSet, self).__init__()
        #self.data = data.sample(len(data), random_state=0).reset_index(drop=True)
        self.data = data
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        # self.A_token = self.tokenizer.encode(text='A', add_special_tokens=False, truncation=True, )
        # self.B_token = self.tokenizer.encode(text='B', add_special_tokens=False, truncation=True, )
        # self.C_token = self.tokenizer.encode(text='C', add_special_tokens=False, truncation=True, )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        now_data = self.data.loc[index]
        idx = now_data['id']
        templete_part1 = "<start_of_turn>user\nHere are two question-answering dialogues. Compare two model performance on answering question, determine which is better.\n\n"
        templete_part1_input_ids = self.tokenizer(text=templete_part1, add_special_tokens=True, padding=False)['input_ids']
        
        templete_part2 = "\n###options\nA. Model A\nB. Model B\nC. Tie\n<end_of_turn>\n"
        templete_part2_input_ids = self.tokenizer(text=templete_part2, add_special_tokens=True, padding=False)['input_ids'][1:]
        #print(f"templete_part2 is {templete_part2_input_ids}")
        templete_part3 = "<start_of_turn>model\n"
        templete_part3_input_ids = self.tokenizer(text=templete_part3, add_special_tokens=True, padding=False)['input_ids'][1:]
        prompt_response = now_data['prompt_response']
        #print(f"id is {now_data['id']}")
        #print(prompt_response)
        prompt_response_ids = self.tokenizer(text=prompt_response, add_special_tokens=True, truncation=True,
                                          max_length=self.max_source_length, padding=False)['input_ids'][1:]
        
        input_ids = templete_part1_input_ids + prompt_response_ids + templete_part2_input_ids + templete_part3_input_ids
        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        #print(f"input is {self.tokenizer.decode(input_ids)}")
        return {
            "input_ids": input_text,
            "id": idx
        }

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in ('input_ids','id')}
    #print(batch)
    batch_input = tokenizer(
        batch['input_ids'],
        padding='longest',
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=MAX_LENGTH + 50
    )
    return batch_input, batch['id']


ex1 = pd.read_csv("dataset/kaggle-ultrafeedback-drop-duplicate.csv")
ex2 = pd.read_csv("dataset/kaggle-ultrafeedback-ties-drop-duplicate.csv")

ex1 = load_json(ex1)
ex2 = load_json(ex2)

data = pd.concat([ex1,ex2]).reset_index(drop = True)

data['id'] = [i for i in range(len(data))]

data.to_json("dataset/ex70k.json", index=False)

data_path = "dataset/ex70k.json"
prompt_type = 2
MAX_INPUT = 2300
if_train = True
df_train , df_valid = load_split_data(data_path, prompt_type, MAX_INPUT, if_train, False, False)
test = df_train

test['length'] = test['prompt_response'].apply(len)
test = test.sort_values(by = ['length'], ascending = False).reset_index(drop = True)

from tqdm import tqdm
def inference(model, test_dataloader):
    test_predictions = []
    for batch in tqdm(test_dataloader):
        batch_input, idx = batch
        for k in batch_input.keys():
            batch_input[k] = batch_input[k].to(device)
        with torch.no_grad():
            response = model.generate(**batch_input, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
            #batch_input['input_ids'].shape[-1] + 1
            score = response.scores[0]
            A_prob, B_prob, C_prob = score[:,A_TOKEN_IDS], score[:,B_TOKEN_IDS], score[:,C_TOKEN_IDS]
            logits = torch.cat([A_prob, B_prob, C_prob], dim=-1) / 1.1
            #logits = torch.Tensor([[A_prob,B_prob,C_prob]]) / 1.1
            logits = torch.softmax(logits, dim=-1).cpu().numpy()
            node_result = [[idx[i],logits[i]] for i in range(len(idx))]
        test_predictions.extend(node_result)
    return test_predictions

device = torch.device("cuda:0")
base_model = 'google/gemma-2-9b-it'
model_path = "output/restful-spaceship-414/checkpoint-5000_896"
MAX_LENGTH = 2300

tokenizer = AutoTokenizer.from_pretrained(model_path, truncation_side = 'left')
config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
base_model_0 = AutoModelForCausalLM.from_pretrained(base_model,
                                                 config=config,
                                                 quantization_config=bnb_config,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto",
                                                 trust_remote_code=True)
# base_model_0.config.pad_token_id = tokenizer.pad_token_id
# base_model_0.resize_token_embeddings(len(tokenizer))
new_model = model_path
model0 = PeftModel.from_pretrained(base_model_0, new_model).to(device)
#model0 = model0.merge_and_unload()
model0.eval()

A_TOKEN_IDS = tokenizer('A',add_special_tokens=True, truncation=True, max_length=1024)['input_ids'][1:]
B_TOKEN_IDS = tokenizer('B',add_special_tokens=True, truncation=True, max_length=1024)['input_ids'][1:]
C_TOKEN_IDS = tokenizer('C',add_special_tokens=True, truncation=True, max_length=1024)['input_ids'][1:]

batch_size = 12
tokenized_dataset = InstructionDataSet(test, tokenizer, MAX_LENGTH, 1)

test_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size = batch_size ,collate_fn=collate_fn)

sub_pred = inference(model = model0, test_dataloader = test_dataloader)

# 提取数据
processed_data = []
for item in sub_pred:
    id = item[0].item()  # 获取id
    array_values = item[1].tolist()  # 获取array并转换为列表
    processed_data.append([id] + array_values)
new_columns = ['id', 'winner_model_a', 'winner_model_b', 'winner_tie']
df = pd.DataFrame(processed_data, columns=new_columns)
df = df.groupby('id').mean().reset_index()

df.to_csv("dataset/prediction.csv", index=False)