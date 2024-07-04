import pandas as pd, numpy as np
import json
from tqdm import tqdm

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
    

def load_split_data(data_path, prompt_type, max_length, if_train, split):
    """
    prompt_type: [1, 2, 3]
    if_train: True or False
    """
    data = pd.read_csv(data_path)

    #safely load data
    data = load_json(data)

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
    if split:
        #split train and valid
        idx = data.id.unique()
        valid_idx = [idx[i] for i in range(len(idx)) if i % 20 == 0]
    
        valid = data.loc[data.id.isin(valid_idx),].reset_index(drop = True)
        train = data.loc[~data.id.isin(valid_idx),].reset_index(drop = True)
    
        return train, valid
    return data, None

    
    

