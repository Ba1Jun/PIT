import os
import logging
import torch
import random
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm


def prepare_plm_data_path(args):
    # chinese
    if args.encoder_type == "cbert":
        args.plm_name = 'bert-base-chinese'
        args.plm_path = '/home/baijun/workspace/models/bert-base-chinese'
    

def prepare_tokenizer(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path, do_lower_case=True, use_fast=True)
    
    return tokenizer


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tokenizer = prepare_tokenizer(args)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    
    def tokenize(self, texts, text_type):
        tokens = []
        input_ids = []
        for text in tqdm(texts, desc=f'[tokenize {text_type}]', leave=True):
            tokens.append(self.tokenizer.tokenize(str(text)))
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens[-1]))
        return tokens, input_ids
    
    def padding_base(self, input_ids_1, input_ids_2, max_len_1, max_len_2, token_type_id_1, token_type_id_2):
        input_ids_1 = list(input_ids_1)
        input_ids_2 = list(input_ids_2)
        _input_ids = []
        token_type_ids = []
        for inst_1, inst_2 in zip(input_ids_1, input_ids_2):
            if max_len_1 == -1:
                _input_ids.append([self.cls_token_id] + inst_1[:254] + [self.sep_token_id] + inst_2[:254] + [self.sep_token_id])
                token_type_ids.append([token_type_id_1] * (len(inst_1[:254])+2) + [token_type_id_2] * (len(inst_2[:254])+1))
            else:
                _input_ids.append([self.cls_token_id] + inst_1[:max_len_1] + [self.sep_token_id] + inst_2[:max_len_2] + [self.sep_token_id])
                token_type_ids.append([token_type_id_1] * (len(inst_1[:max_len_1])+2) + [token_type_id_2] * (len(inst_2[:max_len_2])+1))

        max_len = max([len(s) for s in _input_ids])
        input_ids = np.array([inst + [self.pad_token_id] * (max_len - len(inst)) for inst in _input_ids], dtype=np.int32)
        attention_mask = np.array([[1] * len(inst) + [self.pad_token_id] * (max_len-len(inst)) for inst in _input_ids], dtype=np.int32)
        token_type_ids = np.array([inst + [self.pad_token_id] * (max_len - len(inst)) for inst in token_type_ids], dtype=np.int32)
        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        token_type_ids = torch.LongTensor(token_type_ids)

        return input_ids.to(self.device), attention_mask.to(self.device), token_type_ids.to(self.device)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.args, 
                self.input_ids_1[idx],
                self.input_ids_2[idx],
                self.labels[idx])
    
    def collate_fn(self, raw_batch):
        args = raw_batch[-1][0]
        batch = dict()
        _, input_ids_1, input_ids_2, label = list(zip(*raw_batch))
        batch["input_ids_1"], batch["attention_mask_1"], batch["token_type_ids_1"] \
            = self.padding_base(input_ids_1, input_ids_2, \
                -1 if self.split == "test" else self.args.max_len_1, -1 if self.split == "test" else self.args.max_len_2,\
                    0, 1)
        batch["input_ids_2"], batch["attention_mask_2"], batch["token_type_ids_2"] \
            = self.padding_base(input_ids_2, input_ids_1, \
                -1 if self.split == "test" else self.args.max_len_2, -1 if self.split == "test" else self.args.max_len_1,\
                    1, 0)
        
        batch["label"] = torch.LongTensor(label).to(self.device)
        
        return batch



class QEDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super(QEDataset, self).__init__(args)
        self.split = split
        self.dataset = args.dataset
        self.data_file = f"data/{self.dataset}/{split}.csv"
        self.process()

    def process(self):
        # Load data features from cache or datas et file
        cached_dir = "./cached_data/"
        if not os.path.exists(cached_dir):
            os.makedirs(cached_dir)
        plm_name = [s for s in self.args.plm_path.split('/') if s !=''][-1]
        cached_dataset_file = os.path.join(cached_dir, f"{self.dataset}_{self.split}_{plm_name}")
        # load processed dataset or process the original dataset
        if os.path.exists(cached_dataset_file) and not self.args.overwrite_cache:
            logging.info("Loading dataset from cached file %s", cached_dataset_file)
            data_dict = torch.load(cached_dataset_file)
            self.input_ids_1 = data_dict["sentence1"]
            self.input_ids_2 = data_dict["sentence2"]
            self.labels = data_dict["label"]
        else:
            logging.info("Creating instances from dataset file at %s", self.data_file)
            raw_data = pd.read_csv(self.data_file, sep=',')
            sentences_1 = raw_data['sentence1'].tolist()
            sentences_2 = raw_data['sentence2'].tolist()
            self.labels = raw_data['label'].tolist()
            # tokenize
            sentence_tokens_1, self.input_ids_1 = self.tokenize(sentences_1, 'sentence1')
            sentence_tokens_2, self.input_ids_2 = self.tokenize(sentences_2, 'sentence2')
            
            logging.info(f"sentence-1: {sentences_1[0]}")
            logging.info(f'sentence-1 tokens: {sentence_tokens_1[0]}')
            logging.info(f'sentence-1 input ids: {self.input_ids_1[0]}')
            logging.info('')
            logging.info(f"sentence-2: {sentences_2[0]}")
            logging.info(f'sentence-2 tokens: {sentence_tokens_2[0]}')
            logging.info(f'sentence-2 input ids: {self.input_ids_2[0]}')
            logging.info('')
            # save data
            saved_data = {
                'sentence1': self.input_ids_1,
                'sentence2': self.input_ids_2,
                "label": self.labels
            }
            logging.info("Saving processed dataset to %s", cached_dataset_file)
            torch.save(saved_data, cached_dataset_file)
