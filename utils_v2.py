import pandas as pd, numpy as np
import json
from tqdm import tqdm
tqdm.pandas(desc = 'pandas bar')
from sklearn.model_selection import train_test_split
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

def process(input_str):
    return json.loads(input_str)

def get_label(row):
    label = [idx for idx, option in enumerate(['winner_model_a','winner_model_b','winner_tie']) if row[option] == 1]
    if label[-1] == 0:
        return 'A'
    elif label[-1] == 1:
        return 'B'
    else:
        return 'C'
    return label[-1]

def load_json(data):
    data.loc[:, 'prompt'] = data['prompt'].apply(process)
    data.loc[:, 'response_a'] = data['response_a'].apply(process)
    data.loc[:, 'response_b'] = data['response_b'].apply(process)
    return data

def adjust_values(A, B, a_space, b_space, ex_space):
    # 计算A和a_space的差值
    a_diff = a_space - A
    b_diff = b_space - B
    
    # 第一种情况：A小于a_space，B小于b_space
    if A < a_space and B < b_space:
        ex_space += a_diff + b_diff
        return A, B, ex_space

    # 第二种情况：如果A和B都各自大于自己的space
    elif A > a_space and B > b_space:
        total_extra_needed = (A - a_space) + (B - b_space)
        if total_extra_needed > ex_space:
            A = int(a_space + ex_space / 2)
            B = int(b_space + ex_space / 2)
            ex_space = 0
        else:
            a_space = A
            b_space = B
            ex_space -= total_extra_needed
            
        return A, B, ex_space
        
    # 第三种情况：A或者B其中有一个大于a_space, b_space
    elif A >= a_space or B >= b_space:
        # 如果A大于a_space但是B小于b_space
        if A >= a_space and B <= b_space:
            extra_needed = A - a_space
            ex_space += b_space - B
            #够用
            if ex_space >= extra_needed:
                ex_space -= extra_needed
                
            else:
                #不够用
                #b_space = B + available_space
                A = a_space + ex_space
                ex_space = 0

        # 如果B大于b_space但是A小于a_space
        elif B > b_space and A < a_space:
            extra_needed = B - b_space
            ex_space += a_space - A
            
            if ex_space >= extra_needed:
                ex_space -= extra_needed
                
            else:
                B = b_space + ex_space
                ex_space = 0

        return A, B, ex_space
    

def adjust(current_lengths, prompt_length_space=300, response_length_space=800):
    prompt_length = current_lengths[0]
    response_a_length = current_lengths[1]
    response_b_length = current_lengths[2]
    #先看prompt的额度
    ex_space = max(0, prompt_length_space - prompt_length)
    response_a_length, response_b_length, ex_space = adjust_values(response_a_length, response_b_length, response_length_space, response_length_space, ex_space)
    prompt_length = min(prompt_length, prompt_length_space)
    prompt_length += ex_space

    return prompt_length, response_a_length, response_b_length

def over_max_length(prompt_input_ids, model_a_input_ids, model_b_input_ids, max_length):
    '''
    单条超出max length
    '''
    length = [len(prompt_input_ids), len(model_a_input_ids), len(model_b_input_ids)]
    prompt_length = int(max_length // 5)
    response_length = int((max_length - prompt_length) // 2)
    prompt_max_length, a_max_length, b_max_length = adjust(length, prompt_length, response_length)
    prompt_ids = prompt_input_ids[:prompt_max_length]
    model_a_input_ids = model_a_input_ids[:a_max_length]
    model_b_input_ids = model_b_input_ids[:b_max_length]
    prompt_response_ids = use_in_prompt_1 + prompt_ids + use_in_prompt_2 + model_a_input_ids + use_in_prompt_3 + model_b_input_ids
    return prompt_response_ids

    
def prompt_3(data, max_length, if_train):
    data = data.iloc[::-1].reset_index(drop = True)#反转
    prompt_response = []
    ids = []
    labels = []
    #只有一种可能会超出max length：
    #单条的prompt和reponse加在一起超出max length
    
    text_length = 0
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        text = use_in_prompt_1 + row['prompt_input_ids'] + use_in_prompt_2 + row['response_a_input_ids'] + use_in_prompt_3 + row['response_b_input_ids']
        response_a = row['response_a_input_ids']
        response_b = row['response_b_input_ids']
        prompt = row['prompt_input_ids']
        id = row['id']
        
        if if_train:
            label = row['label_ids']
        
        if id not in ids:
            #第一次出现
            text_length = len(text)
            ids.append(id)
            if if_train:
                labels.append(label)
            if text_length > max_length:
                text = over_max_length(prompt_input_ids = prompt, model_a_input_ids = response_a, model_b_input_ids = response_b, max_length = max_length)
                
            prompt_response.append(text)
        
        else:
            text_length += len(text)
            if text_length <= max_length:
                #取上一个text出来，合并后替换
                text = text + templete_part4_input_ids + prompt_response[-1]
                prompt_response[-1] = text
                
            else:
                #另一起一行
                text_length = len(text)
                ids.append(id)
                
                if if_train:
                    labels.append(label)
                    
                #另起一行但超出长度
                if text_length > max_length:
                    text = over_max_length(prompt_input_ids = prompt, model_a_input_ids = response_a, model_b_input_ids = response_b, max_length = max_length)

                prompt_response.append(text)
                    
                
                    
    if if_train:           
        data = pd.DataFrame({'id': ids, 'prompt_response': prompt_response, "label": labels})
        data = data.iloc[::-1].reset_index(drop = True)#反转
    else:
        data = pd.DataFrame({'id': ids, 'prompt_response': prompt_response})
        data = data.iloc[::-1].reset_index(drop = True)#反转
    return data

def make_prompt(data, if_train, prompt_type, max_length, tokenizer):
    #seperate prompt-response
    data = data.explode(['prompt','response_a','response_b']).reset_index(drop = True)
    
    #prepare label
    if if_train:
        data['label'] = data.apply(lambda x: get_label(x), axis = 1)
        
    '''
    单个id有超过两条以上的对话，就将都为None的丢掉
    否则保留
    '''
    idx = data.response_b.isna() & data.response_a.isna()#都为none
    t = data[idx].id.unique()
    all_none = data.loc[data.id.isin(t)].reset_index(drop = True)
    count = all_none.id.value_counts().reset_index()
    drop_id = count.loc[count['count']>1].id.unique()
    filter_idx = data.id.isin(drop_id) & idx
    data = data[~filter_idx].reset_index(drop = True)
    
    data = data.fillna('None')
    data['response_a'] = data['response_a'].apply(lambda x: 'None' if len(x)==0 else x)
    data['response_b'] = data['response_b'].apply(lambda x: 'None' if len(x)==0 else x)
    
    #分词
    data = data.progress_apply(lambda x: tokenize(x, tokenizer), axis = 1)
    
    if prompt_type == 1:
        data = prompt_1(data)
    elif prompt_type == 2:
        data = prompt_2(data, max_length * 0.75, if_train)
    elif prompt_type == 3:
        data = prompt_3(data, max_length, if_train)
    return data

def tokenize(row, tokenizer):

    now_data = row
    response_a = row['response_a']
    response_a_input_ids = tokenizer(text=response_a, add_special_tokens=False, padding=False)['input_ids']
    row['response_a_input_ids'] = response_a_input_ids
    
    response_b = row['response_b']
    response_b_input_ids = tokenizer(text=response_b, add_special_tokens=False, padding=False)['input_ids']
    row['response_b_input_ids'] = response_b_input_ids
    
    prompt = row['prompt']
    prompt_input_ids = tokenizer(text=prompt, add_special_tokens=False, padding=False)['input_ids']
    row['prompt_input_ids'] = prompt_input_ids
    
    label = now_data['label']
    label_ids = tokenizer.encode(text=label, add_special_tokens=False)
    row['label_ids'] = label_ids

    return row

def load_split_data(data_path, prompt_type, max_length, if_train, split, split_by_prompt, if_drop_duplicate, keep, MODEL):
    """
    prompt_type: [1, 2, 3]
    if_train: True or False
    """
    if "csv" in data_path:
        data = pd.read_csv(data_path)
        #safely load data
        data = load_json(data)
    elif "json" in data_path:
        data = pd.read_json(data_path)

    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, truncation_side = 'left')

    global use_in_prompt_1
    global use_in_prompt_2
    global use_in_prompt_3
    use_in_prompt_1 = tokenizer(text="#Prompt\n", add_special_tokens=False, padding=False)['input_ids']
    use_in_prompt_2 = tokenizer(text="\n\n" + "#Response\n" + "##Model A\n", add_special_tokens=False, padding=False)['input_ids']
    use_in_prompt_3 = tokenizer(text="\n\n" + "##Model B\n", add_special_tokens=False, padding=False)['input_ids']

    global templete_part4_input_ids
    global eos_token_id
    templete_part4_input_ids = tokenizer(text="\n\n", add_special_tokens=False, padding=False)['input_ids']
    eos_token_id = tokenizer.eos_token_id

    if split:
        #split train and valid
        idx = data.id.unique()
        valid_idx = [idx[i] for i in range(len(idx)) if i % 20 == 0]
        
        valid = data.loc[data.id.isin(valid_idx),].reset_index(drop = True)
        train = data.loc[~data.id.isin(valid_idx),].reset_index(drop = True)
            
        train = make_prompt(train, if_train, prompt_type, max_length, tokenizer)
        valid = make_prompt(valid, if_train, prompt_type, max_length, tokenizer)
        if if_drop_duplicate:
            train = train.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
            valid = train.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
        return train, valid
        
    data = make_prompt(data, if_train, prompt_type, max_length, tokenizer)
    if if_drop_duplicate:
            data = data.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
    return data, None

    
    

