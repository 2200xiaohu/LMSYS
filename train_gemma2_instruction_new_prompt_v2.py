import wandb
import os
import yaml
# 使用你的API键进行身份验证
#wandb.login(key="0dc3b3b0446b871143ef4993c923d3e32da9033a")
#0dc3b3b0446b871143ef4993c923d3e32da9033a
#os.environ['WANDB_API_KEY'] = "c465dd55c08ec111e077cf0454ba111b3a764a78"
from transformers import Trainer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from utils_v2 import load_split_data

#os.environ["CUDA_VISIBLE_DEVICES"]="0“

class AWP:
    def __init__(self, model, adv_param="weight", adv_lr=0.1, adv_eps=0.0001):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}


    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 在损失函数之前获得梯度
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

TRAINING_ARGS_NAME = "traning_args.bin"
                
class CustomTrainer(Trainer):
    def __init__(self, 
                model = None,
                args = None,
                data_collator = None,
                train_dataset = None,
                eval_dataset = None,
                tokenizer = None,
                model_init = None,
                compute_metrics = None,
                callbacks = None,
                optimizers = (None, None),
                preprocess_logits_for_metrics = None,
                awp_lr = 0, 
                awp_eps = 0, 
                awp_start_epoch = 0):
        
        super().__init__(model = model,
                        args = args,
                        data_collator = data_collator,
                        train_dataset = train_dataset,
                        eval_dataset = eval_dataset,
                        tokenizer = tokenizer,
                        model_init = model_init,
                        compute_metrics = compute_metrics,
                        callbacks = callbacks,
                        optimizers = optimizers,
                        preprocess_logits_for_metrics = preprocess_logits_for_metrics)
        
        self.awp_lr = awp_lr
        self.awp_eps = awp_eps
        self.awp_start_epoch = awp_start_epoch
        
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     labels = inputs.get("labels")
    #     inputs.pop("labels")  # Remove labels from inputs as model doesn't expect it
    #     #print(f"inputs is {inputs.input_ids.shape}")
    #     #print(f"labels is {labels}")
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")

    #     #print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
    #     #print(f"Logits {logits}, Labels: {labels}")

    #     # Use CrossEntropyLoss for classification
    #     vocab_size = logits.shape[-1]
    #     loss_fct = nn.CrossEntropyLoss()
    #     logits = logits[:,:-1,:]
    #     labels = labels[:,1:]
    #     print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
    #     loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
    #     if loss < 0.1:
    #         print(f"Logits {logits}, Labels: {labels}")
    #     return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving model checkpoint to {output_dir}")
        model_to_save = self.model
        state_dict = {k: v.to("cpu") for k, v in model_to_save.named_parameters() if v.requires_grad}
        # Using Hugging Face's save_pretrained instead of PyTorch's torch.save
        model_to_save.save_pretrained(output_dir, state_dict=state_dict, save_function=torch.save,
                                      safe_serialization=False)

        # Save tokenizer and training arguments as usual
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        print(self.args)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME, ))
        
    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        o_inputs = inputs.copy()
        #inputs = self._prepare_inputs(inputs)
        inputs = self._prepare_inputs(inputs)
      #  print('---'*60)
      #  print(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
            
        ######################## 
        ### AWP 
        if self.awp_lr != 0 and self.state.epoch >= self.awp_start_epoch:
           # print(inputs)
           # print('Start amp')
            self.awp = AWP(model, adv_lr=self.awp_lr, adv_eps=self.awp_eps)
            self.awp._save()
            self.awp._attack_step()
            with self.compute_loss_context_manager():
                awp_loss = self.compute_loss(self.awp.model, o_inputs)
                
            if self.args.n_gpu > 1:
                awp_loss = awp_loss.mean()  # mean() to average on multi-gpu parallel training 
                
            # if self.do_grad_scaling:
            #     self.scaler.scale(awp_loss).backward()
            elif self.use_apex:
                with amp.scale_loss(awp_loss, self.optimizer) as awp_scaled_loss:
                    awp_scaled_loss.backward()
            else:
                self.accelerator.backward(awp_loss)
            self.awp._restore()
        ########################
        
        return loss.detach() / self.args.gradient_accumulation_steps

    
    

import argparse
from typing import Optional, Union

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass

import datasets
from datasets import Dataset

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
    prepare_model_for_kbit_training
)
import os

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
        
        templete_part1 = "<start_of_turn>user\nHere are two question-answering dialogues. Compare two model performance on answering question, determine which is better.\n\n"
        templete_part1_input_ids = self.tokenizer(text=templete_part1, add_special_tokens=True, padding=False)['input_ids']
        
        templete_part2 = "\n###options\nA. Model A\nB. Model B\nC. Tie\n<end_of_turn>\n"
        templete_part2_input_ids = self.tokenizer(text=templete_part2, add_special_tokens=False, padding=False)['input_ids']

        templete_part3 = "<start_of_turn>model\n"
        templete_part3_input_ids = self.tokenizer(text=templete_part3, add_special_tokens=False, padding=False)['input_ids']
        
        templete_part4_input_ids = self.tokenizer(text="\n\n", add_special_tokens=False, padding=False)['input_ids']
        
        prompt_response_ids = now_data['prompt_response']
            
        label_ids = now_data['label']
        input_ids = templete_part1_input_ids + prompt_response_ids + templete_part2_input_ids + templete_part3_input_ids + label_ids + [self.tokenizer.eos_token_id]
        labels = [-100] * (len(input_ids) - 2) + label_ids + [self.tokenizer.eos_token_id]
        print(f"input is {self.tokenizer.decode(input_ids)}")
        return {
            "input_ids": input_ids,
            "labels": labels
        }

@dataclass
class DataCollatorForInstruction:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        #print(f"label_pad_token_id is {self.label_pad_token_id}")
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            # print(padding_side)
            for feature in features:
                # print("before", feature['labels'])
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                # print("after", feature['labels'])
                # print("-" * 60)
        # breakpoint()
        features = self.tokenizer.pad(
            features,
            padding='longest',
            #max_length=max_label_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        #print(f"features is {features}")
        #print(f"labels is {features['labels']}")
        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        # breakpoint() # [(len(features[i]['input_ids']),len(features[i]['labels'])) for i in range(4)]
        #print(features['input_ids'])
        # if self.tokenizer.pad_token_id in features['input_ids']:#
        #     print(f"use padding")
        #     #idx = features['input_ids'].index(128256)
        #     #print(f"padding on: {features['input_ids'][idx-30,: idx+30]}")
        return features
#[num, seq_length, vocab_size]    
#generaate : [num, max_token, vocab_size]
def compute_metrics(p):
    logits = p.predictions
    #print(f"logits before shape is {logits.shape}")
    #predictions = np.argmax(logits, axis=-1)
    # A_prob, B_prob, C_prob = logits[:,0,A_TOKEN_IDS].squeeze(1), logits[:,0,B_TOKEN_IDS].squeeze(1), logits[:,0,C_TOKEN_IDS].squeeze(1)
    # #print(f"A_prob {A_prob.shape}")
    # logits = torch.Tensor([[A_prob,B_prob,C_prob]])
    # logits = torch.softmax(logits, dim=-1)
    logits = logits[:,0,[A_TOKEN_IDS,B_TOKEN_IDS,C_TOKEN_IDS]]
    logits = torch.softmax(torch.tensor(logits).reshape(-1,3), dim=-1)
    # print(f"logits shape is {logits.shape}")
    # print(f"logits is {logits}")
    labels = torch.tensor(p.label_ids)
    # print(f"labels is {labels}")
    # print(f"labels shape is {labels.shape}")

    # get first none -100
    bs, seq_len = labels.shape
    mask = labels != -100
    _, indices = torch.max(mask, dim=1)
    
    row_indices = torch.arange(bs).unsqueeze(1)
    col_indices = (indices.unsqueeze(1) + torch.arange(1)).clamp(max=seq_len-1)

    labels = labels[row_indices, col_indices]
    
    token2num = {A_TOKEN_IDS[0]:0, B_TOKEN_IDS[0]:1, C_TOKEN_IDS[0]:2}
    labels = labels.reshape(-1).numpy()
    labels = np.array([token2num.get(val, val) for val in labels])
    # print(f"labels is {labels}")
    # print(f"labels shape is {labels.shape}")
    prediction = logits.tolist()
    # print(f"prediction shape is {prediction}, len {len(prediction)}")
    return {"log_loss": log_loss(labels, prediction, labels=[0,1,2])}

class SaveModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "sft_lora_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
        kwargs["model"].save_pretrained(peft_model_path)
        kwargs["tokenizer"].save_pretrained(peft_model_path)

# def process(input_str):
#     if len(input_str) < 10:
#         return 'None'
    
#     else:
#         stripped_str = input_str.strip('[]')
#         sentences = [s.strip('"') for s in stripped_str.split('","')]
#         return  ' '.join(sentences)

# def process(input_str):
#     return json.loads(input_str)
# import json
# def load_json(data, all_in_one):
#     if all_in_one:
#         data.loc[:, 'prompt_response'] = data['prompt_response'].apply(process)
#         return data
#     else:
#         data.loc[:, 'prompt'] = data['prompt'].apply(process)
#         data.loc[:, 'response_a'] = data['response_a'].apply(process)
#         data.loc[:, 'response_b'] = data['response_b'].apply(process)
#         return data

def preprocess_logits_for_metrics(logits, labels):
    # print(logits.shape)
    # print(labels.shape)
    # print(logits)
    # print(labels)
    # logits = logits.cpu()
    # labels = labels.cpu()
    bs, seq_len, vocab_size = logits.shape
    mask = labels != -100
    _, indices = torch.max(mask, dim=1)
    indices = indices - 1
    row_indices = torch.arange(bs).unsqueeze(1).cuda()
    last_two = torch.arange(2).cuda()
    col_indices = (indices.unsqueeze(1) + last_two).clamp(max=seq_len-1).cuda()
    logits = logits[row_indices, col_indices,:]
    # print(f"logits.shape is {logits.shape}")
    return logits
    
def train(args):
    # set the wandb project where this run will be logged
    # os.environ["WANDB_PROJECT"]=args.output_dir
    from time import gmtime, strftime
 
    # using simple format of showing time
    s = strftime("%a_%d_%b_%H_%M", gmtime())

    wandb.init(project="LMSYS", config=args)
    wandb.save("train_gemma2_instruction_new_prompt_v2.py")
    wandb.save("train_gemma2_instruction_new_prompt_v2.yaml")
    # HUGGING FACE MODEL
    MODEL = args.MODEL

    if len(args.train_data) != 0:
        #加载基本数据集
        print(f"loading base train data: {args.train_data}")
        df_train, _ = load_split_data(args.train_data, args.prompt_type, args.MAX_INPUT, args.if_train, False, False, args.if_drop_duplicate, args.keep, MODEL)
    if len(args.valid_data) != 0: 
        print(f"loading base valid data: {args.valid_data}")
        df_valid, _ = load_split_data(args.valid_data, args.prompt_type, args.MAX_INPUT, args.if_train, False, False, args.if_drop_duplicate, args.keep, MODEL)
        
    if len(args.data_path) !=0:    
        ### load data
        print(f"loading extrnal data")
        ex_train = pd.DataFrame()
        for p in args.data_path:
            print(f"extrnal data {p}")
            tmp_train , _ = load_split_data(p, args.prompt_type, args.MAX_INPUT, args.if_train, False, False, args.if_drop_duplicate, args.keep, MODEL)
            ex_train = pd.concat([ex_train,tmp_train]).reset_index(drop = True)
        #df_train = pd.read_csv(args.train_data).reset_index(drop = True)
        #df_valid = pd.read_csv(args.valid_data).reset_index(drop = True)
    
        # df_train = load_json(df_train, args.all_in_one)
        # df_valid = load_json(df_valid, args.all_in_one)
        #df_train = df_train.loc[:500,:].reset_index(drop = True)
    
        # else:
        #     df_valid = df_valid.loc[:500,:].reset_index(drop = True)
        if args.extranal_data == True:
            #得到原有的验证集
            df_valid, _ = load_split_data('dataset/non_overlap/valid.json', args.prompt_type, args.MAX_INPUT, args.if_train, False, False, args.if_drop_duplicate, args.keep, MODEL)
            if args.if_concat:
                print(f"concat extrnal data and base train data")
                df_train = pd.concat([df_train, ex_train]).reset_index(drop = True)
            else:
                print(f"only use extrnal data")
                df_train = ex_train
                
    if args.test_mode:
        df_valid = df_valid.loc[:20,:].reset_index(drop = True)
        
    df_train = df_train.sample(len(df_train)).reset_index(drop = True)
    # df_train.loc[:, 'prompt'] = df_train['prompt'].apply(process)
    # df_train.loc[:, 'response_a'] = df_train['response_a'].apply(process)
    # df_train.loc[:, 'response_b'] = df_train['response_b'].apply(process)
    
    # df_valid.loc[:, 'prompt'] = df_valid['prompt'].apply(process)
    # df_valid.loc[:, 'response_a'] = df_valid['response_a'].apply(process)
    # df_valid.loc[:, 'response_b'] = df_valid['response_b'].apply(process)
    
    ### process dataset 
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, truncation_side = 'left')
    #tokenizer.add_special_tokens({"pad_token":"<pad>"})
    
    print(f"arg is {args}")
    
    train_dataset_path = './dataset_cache/' + args.train_data.split('/')[-1].split('.')[0] + '_' + args.MODEL.replace('/','-') + '_' + args.token_type
    valid_dataset_path = './dataset_cache/' + args.valid_data.split('/')[-1].split('.')[0] + '_' + args.MODEL.replace('/','-') + '_' + args.token_type
    if not os.path.exists(train_dataset_path):
        os.makedirs(train_dataset_path)
    if not os.path.exists(valid_dataset_path):
        os.makedirs(valid_dataset_path)
    train_cache_path = os.path.join(train_dataset_path, 'dataset.bin')
    valid_cache_path = os.path.join(valid_dataset_path, 'dataset.bin')
    if args.use_cache and os.path.exists(train_cache_path):
        tokenized_dataset = torch.load(train_cache_path)
    else:
        tokenized_dataset = InstructionDataSet(df_train,tokenizer, args.MAX_INPUT, 1)
        torch.save(tokenized_dataset, train_cache_path)
    print(args.use_cache)
    if args.use_cache and os.path.exists(valid_cache_path):
        tokenized_dataset_valid = torch.load(valid_cache_path)
    else:
        tokenized_dataset_valid = InstructionDataSet(df_valid,tokenizer, args.MAX_INPUT, 1)
        torch.save(tokenized_dataset_valid, valid_cache_path)   

    global A_TOKEN_IDS
    A_TOKEN_IDS = tokenizer('A',add_special_tokens=True, truncation=True, max_length=1024)['input_ids'][1:]
    global B_TOKEN_IDS 
    B_TOKEN_IDS = tokenizer('B',add_special_tokens=True, truncation=True, max_length=1024)['input_ids'][1:]
    global C_TOKEN_IDS 
    C_TOKEN_IDS = tokenizer('C',add_special_tokens=True, truncation=True, max_length=1024)['input_ids'][1:]

    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,  # 使用8bit量化
    #     bnb_8bit_quant_type='nf8',
    #     bnb_8bit_compute_dtype=torch.bfloat16,
    #     bnb_8bit_use_double_quant=False
    # )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(MODEL,
                                                 config=config,
                                                 quantization_config=bnb_config,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map="auto",
                                                 trust_remote_code=True,
                                                 attn_implementation='eager')

    # model.config.pad_token_id = tokenizer.pad_token_id
    # model.resize_token_embeddings(len(tokenizer))
    #model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # config = AutoConfig.from_pretrained(MODEL)
    # config.hidden_dropout_prob = args.dropout_rate
    # config.attention_probs_dropout_prob = args.dropout_rate

    
    checkpoint = None
    if len(args.resume_from_checkpoint) != 0:
        checkpoint = args.resume_from_checkpoint
        print(f"Using checkpoint: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=True)
        print(model)
        
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # For sequence classification
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            #bias = 'none',
            target_modules=['q_proj','k_proj','v_proj','o_proj'] #,
        )
        model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    # for key in model.state_dict():
    #     print(f"{key}, {model.state_dict()[key].shape}, {model.state_dict()[key].dtype}")
        
    data_collator = DataCollatorForInstruction(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False,
    )
    
    training_args = TrainingArguments(
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            output_dir=f"output/{wandb.run.name}",
            report_to="wandb",
            overwrite_output_dir=True,
            greater_is_better = False,
            fp16=False,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            evaluation_strategy='steps',
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_only_model = True,
            load_best_model_at_end=False,
            metric_for_best_model='log_loss',
            lr_scheduler_type='cosine',
            weight_decay=args.weight_decay,
            save_total_limit=30,
            label_smoothing_factor=args.label_smoothing_factor,
            # do_eval=False,
            # evaluation_strategy="no"
        )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

    # scheduler = get_polynomial_decay_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=training_args.warmup_steps,
    #     num_training_steps=training_args.num_train_epochs *
    #     int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /
    #         training_args.gradient_accumulation_steps),
    #     power=1.0,
    #     lr_end=args.lr_end
    # )
    num_warmup_steps = int(args.warmup_ratio * training_args.num_train_epochs * int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /training_args.gradient_accumulation_steps))
    scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps= num_warmup_steps ,
            num_training_steps=training_args.num_train_epochs *
                int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /
                    training_args.gradient_accumulation_steps),
            num_cycles = 0.5)#3
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_valid,
        compute_metrics = compute_metrics,
        optimizers=(optimizer, scheduler),
        awp_lr = args.awp_lr,
        awp_eps = args.awp_eps,
        awp_start_epoch = args.awp_start_epoch,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )


    #model.gradient_checkpointing_enable()
    #model.enable_input_require_grads()
    trainer.train()
    #trainer.add_callback(SaveModelCallback)
    # trainer.save_model(args.output_dir)
    wandb.finish()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Demo of argparse')
    # parser.add_argument('--train_data', default='dataset/random_train.csv', type=str)
    # parser.add_argument('--valid_data', default='dataset/random_valid.csv', type=str)
    # parser.add_argument('--per_device_train_batch_size', default = 1, type=int)
    # parser.add_argument('--per_device_eval_batch_size', default = 1, type=int)
    # parser.add_argument('--learning_rate', default = 4e-6, type=float)
    # parser.add_argument('--lr_end', default = 2e-6, type=float)
    # parser.add_argument('--warmup_ratio', default = 0.1, type=float)
    # parser.add_argument('--num_train_epochs', default = 5, type=int)
    # parser.add_argument('--gradient_accumulation_steps', default = 8, type=int)
    # parser.add_argument('--logging_steps', default = 5, type=int)
    # parser.add_argument('--eval_steps', default = 8, type=int)
    # parser.add_argument('--save_steps', default = 8, type=int)
    # parser.add_argument('--weight_decay', default = 1e-4, type=float)    
    # parser.add_argument('--MAX_INPUT', default = 1024, type=int)
    # parser.add_argument('--MODEL', default='microsoft/deberta-v3-base', type=str)
    # parser.add_argument('--dropout_rate', default = 0.1, type=float) 
    # parser.add_argument('--awp_lr', default = 0.1, type=float)
    # parser.add_argument('--awp_eps', default = 1e-4, type=float)
    # parser.add_argument('--awp_start_epoch', default = 0.5, type=float)
    # parser.add_argument('--label_smoothing_factor', default = 0, type=float)
    # parser.add_argument('--output_dir', default='output', type=str)
    # parser.add_argument('--use_cache', action='store_true')
    # parser.add_argument('--token_type', default = 'MC', type = str)
    # parser.add_argument('--lora_r', default = '64', type = int)
    # parser.add_argument('--lora_alpha', default = '32', type = int)
    # parser.add_argument('--lora_dropout', default = '0.2', type = float)
    # args = parser.parse_args()

    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--config', default='train_gemma2_instruction_new_prompt_v2.yaml', type=str, help='Path to the config file', required=False)
    args = parser.parse_args()

    config = load_config(args.config)

    for key, value in config.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))

    args = parser.parse_args()
    
    train(args)

#python train_llama.py --MODEL meta-llama/Meta-Llama-3-8B-Instruct --eval_steps 200 --save_steps 200 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-5 --gradient_accumulation_steps 16 --awp_lr 0
#--train_data dataset/demo_train.csv --valid_data dataset/demo_valid.csv 
#python train_llama.py