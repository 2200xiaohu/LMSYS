import wandb
import os
# 使用你的API键进行身份验证
#wandb.login(key="0dc3b3b0446b871143ef4993c923d3e32da9033a")
#0dc3b3b0446b871143ef4993c923d3e32da9033a
#os.environ['WANDB_API_KEY'] = "c465dd55c08ec111e077cf0454ba111b3a764a78"
from transformers import Trainer
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
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

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
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
                
            if self.do_grad_scaling:
                self.scaler.scale(awp_loss).backward()
            elif self.use_apex:
                with amp.scale_loss(awp_loss, self.optimizer) as awp_scaled_loss:
                    awp_scaled_loss.backward()
            else:
                self.accelerator.backward(awp_loss)
            self.awp._restore()
        ########################
        
        return loss.detach() / self.args.gradient_accumulation_steps
    

import os

from typing import Optional, Union
import pandas as pd, numpy as np, torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoConfig
from transformers import EarlyStoppingCallback
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, RobertaForMultipleChoice
import argparse
from transformers import get_polynomial_decay_schedule_with_warmup, TrainerCallback
import datasets
from datasets import Dataset
from sklearn.metrics import log_loss
import torch.nn as nn
    
@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        print(f"features is {features}")
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch
    
def compute_metrics(p):
    print(f"predict is {p}")
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"logloss": log_loss(predictions, labels)}

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

    
def train(args):
    # set the wandb project where this run will be logged
    # os.environ["WANDB_PROJECT"]=args.output_dir
    from time import gmtime, strftime
 
    # using simple format of showing time
    s = strftime("%a_%d_%b_%H_%M", gmtime())

    wandb.init(project="LMSYS", config=args)
        
    # HUGGING FACE MODEL
    MODEL = args.MODEL
    
    ### load data
    df_train = pd.read_csv(args.train_data).reset_index(drop = True)
    df_valid = pd.read_csv(args.valid_data).reset_index(drop = True)
    ### process dataset 
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
    index_to_option = {v: k for k,v in option_to_index.items()}
    def preprocess(example):
        first_sentence = [ "[CLS] " + example['prompt'] ] * 2
        second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in ['response_a','response_b']]
        tokenized_example = tokenizer(first_sentence, second_sentences, truncation='longest_first', 
                                      max_length=args.MAX_INPUT, add_special_tokens=False)
        tokenized_example['label'] = example['label']
        return tokenized_example
    
    train_dataset_path = './' + args.train_data.split('/')[-1].split('.')[0] + '_' + args.MODEL.replace('/','-') + '_' + args.token_type
    valid_dataset_path = './' + args.valid_data.split('/')[-1].split('.')[0] + '_' + args.MODEL.replace('/','-') + '_' + args.token_type
    if not os.path.exists(train_dataset_path):
        os.makedirs(train_dataset_path)
    if not os.path.exists(valid_dataset_path):
        os.makedirs(valid_dataset_path)
    train_cache_path = os.path.join(train_dataset_path, 'dataset.bin')
    valid_cache_path = os.path.join(valid_dataset_path, 'dataset.bin')
    if args.use_cache and os.path.exists(train_cache_path):
        tokenized_dataset = torch.load(train_cache_path)
    else:
        dataset = datasets.Dataset.from_pandas(df_train)
        tokenized_dataset = dataset.map(preprocess, remove_columns=['id', 'model_a', 'model_b', 'prompt', 'response_a', 'response_b'])
        torch.save(tokenized_dataset, train_cache_path)
    print(args.use_cache)
    if args.use_cache and os.path.exists(valid_cache_path):
        tokenized_dataset_valid = torch.load(valid_cache_path)
    else:
        dataset_valid = datasets.Dataset.from_pandas(df_valid)
        tokenized_dataset_valid = dataset_valid.map(preprocess, remove_columns=['id', 'model_a', 'model_b', 'prompt', 'response_a', 'response_b'])
        torch.save(tokenized_dataset_valid, valid_cache_path)   

    config = AutoConfig.from_pretrained(MODEL)
    config.hidden_dropout_prob = args.dropout_rate
    config.attention_probs_dropout_prob = args.dropout_rate
    if 'roberta' in MODEL:
        model = RobertaForMultipleChoice.from_pretrained(MODEL,config = config)
    else:
        model = AutoModelForMultipleChoice.from_pretrained(MODEL)
    for key in model.state_dict():
        print(key, model.state_dict()[key].shape)
    
    training_args = TrainingArguments(
            warmup_ratio=args.warmup_ratio, 
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            output_dir = args.output_dir,
            overwrite_output_dir=True,
            fp16=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=args.logging_steps,
            evaluation_strategy='steps',
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            load_best_model_at_end=False,
            metric_for_best_model='logloss',
            lr_scheduler_type='cosine',
            weight_decay=args.weight_decay,
            save_total_limit=3,
            label_smoothing_factor=args.label_smoothing_factor
        )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * training_args.num_train_epochs *
        int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /
            training_args.gradient_accumulation_steps)),
        num_training_steps=training_args.num_train_epochs *
        int(len(tokenized_dataset) * 1.0 / training_args.per_device_train_batch_size /
            training_args.gradient_accumulation_steps),
        power=1.0,
        lr_end=args.lr_end
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_valid,
        compute_metrics = compute_metrics,
        optimizers=(optimizer, scheduler),
        awp_lr = args.awp_lr,
        awp_eps = args.awp_eps,
        awp_start_epoch = args.awp_start_epoch
    )
    
    trainer.train()
    # trainer.add_callback(SaveModelCallback)
    # trainer.save_model(args.output_dir)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--train_data', default='dataset/demo_train.csv', type=str)
    parser.add_argument('--valid_data', default='dataset/demo_valid.csv', type=str)
    parser.add_argument('--per_device_train_batch_size', default = 1, type=int)
    parser.add_argument('--per_device_eval_batch_size', default = 1, type=int)
    parser.add_argument('--learning_rate', default = 4e-6, type=float)
    parser.add_argument('--lr_end', default = 2e-6, type=float)
    parser.add_argument('--warmup_ratio', default = 0.1, type=float)
    parser.add_argument('--num_train_epochs', default = 5, type=int)
    parser.add_argument('--gradient_accumulation_steps', default = 8, type=int)
    parser.add_argument('--logging_steps', default = 5, type=int)
    parser.add_argument('--eval_steps', default = 8, type=int)
    parser.add_argument('--save_steps', default = 8, type=int)
    parser.add_argument('--weight_decay', default = 1e-4, type=float)    
    parser.add_argument('--MAX_INPUT', default = 8, type=int)
    parser.add_argument('--MODEL', default='microsoft/deberta-v3-base', type=str)
    parser.add_argument('--dropout_rate', default = 0.1, type=float) 
    parser.add_argument('--awp_lr', default = 0.1, type=float)
    parser.add_argument('--awp_eps', default = 1e-4, type=float)
    parser.add_argument('--awp_start_epoch', default = 0.5, type=float)
    parser.add_argument('--label_smoothing_factor', default = 0, type=float)
    parser.add_argument('--output_dir', default='output', type=str)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--token_type', default = 'MC', type = str)
    args = parser.parse_args()
    train(args)
