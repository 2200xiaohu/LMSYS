import pandas as pd, numpy as np
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
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

def prompt_1(data):
    '''
    #Model A
    Prompt1: xxx
    Response: xxx

    Prompt2: xxx
    Response: xxx
    
    #Model B
    Prompt1: xxx
    Response: xxx

    Prompt2: xxx
    Response: xxx
    '''
    data['prompt_response_A'] = "Prompt: " + data['prompt'] + "\n" + "Response: " + data['response_a']
    data['prompt_response_B'] = "Prompt: " + data['prompt'] + "\n" + "Response: " + data['response_b']
    data = data.groupby('id').agg({'prompt_response_A': '\n\n'.join, 'prompt_response_B': '\n\n'.join, 'label': lambda x: list(x)[0]}).reset_index()
    data['prompt_response'] = "#Model A\n" + data['prompt_response_A'] + "\n\n#Model B\n" + data['prompt_response_B']
    return data

def prompt_2(data, max_length, if_train):
    '''
    超过max length新开一行，label不变
    #Prompt1
    xxxx
    #Response
    ##Model A
    xxxx
    ##Model B
    xxxx
    
    #Prompt2
    #Response
    ##Model A
    xxxx
    ##Model B
    xxxx
    '''

    data['prompt_response'] = "#Prompt\n" + data['prompt'] + "\n\n" + "#Response\n" + "##Model A\n" + data['response_a'] + "\n\n" + "##Model B\n" + data['response_b']

    prompt_response = []
    ids = []
    labels = []
    text_length = 0
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        text = row['prompt_response']
        if if_train:
            label = row['label']
        id = row['id']
        if id not in ids:
            #第一次出现
            prompt_response.append(text)
            text_length = len(text.split(" "))
            ids.append(id)
            if if_train:
                labels.append(label)
        else:
            text_length += len(text.split(" "))
            if text_length <= max_length:
                #取上一个text出来，合并后替换
                text = prompt_response[-1] + "\n\n" + text
                prompt_response[-1] = text
            else:
                #另一起一行
                prompt_response.append(text)
                text_length = len(text.split(" "))
                ids.append(id)
                if if_train:
                    labels.append(label)
    if if_train:           
        data = pd.DataFrame({'id': ids, 'prompt_response': prompt_response, "label": labels})
    else:
        data = pd.DataFrame({'id': ids, 'prompt_response': prompt_response})
    return data

def get_text_length(text):
    '''
    不用空格分隔的文本, text length = len
    不用空格分隔的一般tokenizer后长度类似，所以还可以缩小
    空格分隔的，len(text.split(" "))
    '''
    length1 = len(text)
    length2 = len(text.split(" "))
    #远超过
    if length1 >= length2 * 30 and length1>= 300:
        return length1 * 0.75
    return length2
    
def prompt_3(data, max_length, if_train):
    '''
    超过max length新开一行，label不变
    从后往前拼接
    #Prompt1
    xxxx
    #Response
    ##Model A
    xxxx
    ##Model B
    xxxx
    
    #Prompt2
    #Response
    ##Model A
    xxxx
    ##Model B
    xxxx
    '''

    data['prompt_response'] = "#Prompt\n" + data['prompt'] + "\n\n" + "#Response\n" + "##Model A\n" + data['response_a'] + "\n\n" + "##Model B\n" + data['response_b']
    data = data.iloc[::-1].reset_index(drop = True)#反转
    prompt_response = []
    ids = []
    labels = []
    #只有一种可能会超出max length：
    #单条的prompt和reponse加在一起超出max length
    over_max_length = [] #是否有超出max length的部分
    overflow_prompt = []
    overflow_response_a = [] #超出max length的部分
    overflow_response_b = [] #超出max length的部分
    text_length = 0
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        text = row['prompt_response']
        response_a = row['response_a']
        response_b = row['response_b']
        prompt = row['prompt']
        id = row['id']
        
        if if_train:
            label = row['label']
        
        if id not in ids:
            #第一次出现
            prompt_response.append(text)
            text_length = get_text_length(text)
            ids.append(id)
            if if_train:
                labels.append(label)
            if text_length > max_length:
                over_max_length.append(1)
                overflow_prompt.append(prompt)
                overflow_response_a.append(response_a)
                overflow_response_b.append(response_b)
            else:
                over_max_length.append(0)
                overflow_prompt.append(None)
                overflow_response_a.append(None)
                overflow_response_b.append(None)
        
        else:
            text_length += get_text_length(text)
            if text_length <= max_length:
                #取上一个text出来，合并后替换
                text = text + "\n\n" + prompt_response[-1]
                prompt_response[-1] = text
                over_max_length[-1] = 0
                overflow_prompt[-1] = None
                overflow_response_a[-1] = None
                overflow_response_b[-1] = None
                
            else:
                #另一起一行
                prompt_response.append(text)
                text_length = get_text_length(text)
                ids.append(id)
                
                if if_train:
                    labels.append(label)
                    
                #另起一行但超出场合都
                if text_length > max_length:
                    over_max_length.append(1)
                    overflow_prompt.append(prompt)
                    overflow_response_a.append(response_a)
                    overflow_response_b.append(response_b)
                else:
                    over_max_length.append(0)
                    overflow_prompt.append(None)
                    overflow_response_a.append(None)
                    overflow_response_b.append(None)
                    
                
                    
    if if_train:           
        data = pd.DataFrame({'id': ids, 'prompt_response': prompt_response, "label": labels, 'overflow_prompt':overflow_prompt, 'over_max_length': over_max_length, 'overflow_response_a': overflow_response_a, 'overflow_response_b': overflow_response_b})
        data = data.iloc[::-1].reset_index(drop = True)#反转
    else:
        data = pd.DataFrame({'id': ids, 'prompt_response': prompt_response, 'over_max_length': over_max_length, 'overflow_prompt':overflow_prompt, 'overflow_response_a': overflow_response_a, 'overflow_response_b': overflow_response_b})
        data = data.iloc[::-1].reset_index(drop = True)#反转
    return data

def make_prompt(data, if_train, prompt_type, max_length):
    #seperate prompt-response
    data = data.explode(['prompt','response_a','response_b']).reset_index(drop = True)

    #prepare label
    if if_train:
        data['label'] = data.apply(lambda x: get_label(x), axis = 1)
    
    data = data.fillna('None')
    data['response_a'] = data['response_a'].apply(lambda x: 'None' if len(x)==0 else x)
    data['response_b'] = data['response_b'].apply(lambda x: 'None' if len(x)==0 else x)
    
    if prompt_type == 1:
        data = prompt_1(data)
    elif prompt_type == 2:
        data = prompt_2(data, max_length * 0.75, if_train)
    elif prompt_type == 3:
        data = prompt_3(data, max_length * 0.75, if_train)
    return data
def load_split_data(data_path, prompt_type, max_length, if_train, split, split_by_prompt, if_drop_duplicate, keep):
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
        
    if split:
        if split_by_prompt:
            # 提取唯一的 prompt 进行划分
            data['prompt_str'] = data['prompt'].astype(str)
            unique_prompts = data['prompt_str'].unique()
            train_prompts, valid_prompts = train_test_split(unique_prompts, test_size=0.1, random_state=42)

            train_prompts_set = set(train_prompts)
            valid_prompts_set = set(valid_prompts)
            
            # 根据划分的 prompt 获取对应的行
            train = data[data['prompt_str'].isin(train_prompts_set)].reset_index(drop = True)
            valid = data[data['prompt_str'].isin(valid_prompts_set)].reset_index(drop = True)
            train = train.drop(columns = ['prompt_str'])
            valid = valid.drop(columns = ['prompt_str'])
            
        else: 
            #split train and valid
            idx = data.id.unique()
            valid_idx = [idx[i] for i in range(len(idx)) if i % 20 == 0]
            
            valid = data.loc[data.id.isin(valid_idx),].reset_index(drop = True)
            train = data.loc[~data.id.isin(valid_idx),].reset_index(drop = True)
            
        train = make_prompt(train, if_train, prompt_type, max_length)
        valid = make_prompt(valid, if_train, prompt_type, max_length)
        if if_drop_duplicate:
            train = train.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
            valid = train.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
        return train, valid
        
    data = make_prompt(data, if_train, prompt_type, max_length)
    if if_drop_duplicate:
            data = data.drop_duplicates(subset = ['id'], keep =keep).reset_index(drop = True)
    return data, None

    
    

