import torch
import sklearn
import numpy as np
import pandas as pd
import time
from typing import Optional, Union

from transformers import AutoTokenizer, LlamaModel, LlamaForSequenceClassification, BitsAndBytesConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
import datasets
from datasets import Dataset
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass
from torch.cuda.amp import autocast
from threading import Thread

import gc
import os
import io
import time
import json
import random
import pickle
import zipfile
import datetime
import matplotlib.pyplot as plt
from IPython.display import display
from collections import Counter
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import log_loss
import tokenizers

import random
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

@dataclass
class DataCollatorForClassification:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        #print(f"features is {features}")
        #label_name = 'label' if 'label' in features[0].keys() else 'labels'
        #labels = [feature.pop(label_name) for feature in features]

        # Flatten the features (no need to handle multiple choices)
        #self.padding = False
        #print(self.padding)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        
        # Adjust the shape of input_ids to ensure [batch_size, sequence_length]
        if batch['input_ids'].dim() == 3:
            batch['input_ids'] = batch['input_ids'].squeeze(1)
        if batch['input_ids'].dim() == 1:
            batch['input_ids'] = batch['input_ids'].unsqueeze(0)

        if 'token_type_ids' in batch:
            if batch['token_type_ids'].dim() == 3:
                batch['token_type_ids'] = batch['token_type_ids'].squeeze(1)
            if batch['token_type_ids'].dim() == 1:
                batch['token_type_ids'] = batch['token_type_ids'].unsqueeze(0)

        if 'attention_mask' in batch:
            if batch['attention_mask'].dim() == 3:
                batch['attention_mask'] = batch['attention_mask'].squeeze(1)
            if batch['attention_mask'].dim() == 1:
                batch['attention_mask'] = batch['attention_mask'].unsqueeze(0)
                
#         # Directly add labels to the batch
#         batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch
    
def preprocess(example):
        sentences = [" # Prompt" + "\n" + example['prompt'] + "\n\n" + "# Answer A" + "\n" + example['response_a'] + "\n\n" +  "# Answer B" + "\n" + example['response_b']]
        #print(f"sentences is {sentences}")
        tokenized_example = tokenizer(sentences, truncation=True, padding='max_length',
                                      max_length=MAX_LENGTH)
        return tokenized_example

test = pd.read_csv('dataset/random_valid.csv')#.sample(5).reset_index(drop = True)
#test = test.loc[:100,:].reset_index(drop=True)
#sample_sub = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/sample_submission.csv')


#concatenate strings in list
def process(input_str):
    if len(input_str) < 10:
        return 'None'
    
    else:
        stripped_str = input_str.strip('[]')
        sentences = [s.strip('"') for s in stripped_str.split('","')]
        return  ' '.join(sentences)


test.loc[:, 'prompt'] = test['prompt'].apply(process)
test.loc[:, 'response_a'] = test['response_a'].apply(process)
test.loc[:, 'response_b'] = test['response_b'].apply(process)

from tqdm import tqdm
def inference(model, test_dataloader):
    test_predictions = []
    for batch in tqdm(test_dataloader):
        for k in batch.keys():
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch)
            #logits = outputs.logits.cpu().detach().numpy()
            predict = torch.softmax(outputs.logits, dim=-1).cpu().numpy()#.to(torch.float)
            #redict = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        test_predictions.append(predict)
    return test_predictions

device = torch.device("cuda:0")

base_model = 'meta-llama/llama-3-transformers-8b-hf-v1'
model_path = "output/warm-breeze-102/checkpoint-1400"
MAX_LENGTH = 1500

tokenizer = AutoTokenizer.from_pretrained(model_path)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
base_model_0 = LlamaForSequenceClassification.from_pretrained(
    base_model,
    num_labels=3,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map=device,
    trust_remote_code=True)
base_model_0.config.pad_token_id = tokenizer.pad_token_id
base_model_0.resize_token_embeddings(len(tokenizer))
new_model = model_path
model0 = PeftModel.from_pretrained(base_model_0, new_model).to(device)
#model0 = model0.merge_and_unload()
model0.eval()

dataset = datasets.Dataset.from_pandas(test)
#['id', 'prompt', 'response_a', 'response_b']
tokenized_dataset = dataset.map(preprocess, remove_columns=test.columns.tolist())

data_collator = DataCollatorForClassification(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

sub_pred = inference(model = model0, test_dataloader = test_dataloader)

prediciton = np.vstack(sub_pred)

new_columns = ['winner_model_a', 'winner_model_b', 'winner_tie']
# 将新的列数据转换为 DataFrame
new_columns_df = pd.DataFrame(prediciton, columns=new_columns)
new_columns_df.to_csv('inference_on_test.csv',index = False)


from sklearn.metrics import log_loss

metric = log_loss(test.label, prediciton)

print(f"log_loss is {metric}")