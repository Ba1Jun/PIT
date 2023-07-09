import json
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel


class MatchingModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        
        self.encoder = BertModel.from_pretrained(args.plm_path)

        self.predictor = nn.Linear(self.encoder.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.pit_weight = float(args.pit_weight)
    
    def predict(self, input_ids, attention_mask, token_type_ids):
        encoder_outputs = self.encoder(input_ids=input_ids, 
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       return_dict=True)
        pooler_output = self.pooler(encoder_outputs, attention_mask)
        predict_logits = self.predictor(pooler_output)
        return predict_logits

    def pooler(self, encoder_outputs, attention_mask):
        last_hidden_state, pooler_output = encoder_outputs['last_hidden_state'], encoder_outputs['pooler_output']
        return last_hidden_state[:, 0]

    def forward(self, input_ids_1, attention_mask_1, token_type_ids_1, input_ids_2=None, attention_mask_2=None, token_type_ids_2=None, label=None):
        # import pdb; pdb.set_trace()
        if self.training:
            input_ids = torch.cat([input_ids_1, input_ids_2], 0)
            attention_mask = torch.cat([attention_mask_1, attention_mask_2], 0)
            token_type_ids = torch.cat([token_type_ids_1, token_type_ids_2], 0)
            
            predict_logits = self.predict(input_ids, attention_mask, token_type_ids)
            predict_logits_1, predict_logits_2 = torch.split(predict_logits, input_ids_1.shape[0], 0)

            kld_loss_1 = self.kld_loss(self.log_softmax(predict_logits_1), self.softmax(predict_logits_2))
            kld_loss_1 = torch.mean(torch.sum(kld_loss_1, dim=1))
            kld_loss_2 = self.kld_loss(self.log_softmax(predict_logits_2), self.softmax(predict_logits_1))
            kld_loss_2 = torch.mean(torch.sum(kld_loss_2, dim=1))
            kld_loss = (kld_loss_1 + kld_loss_2) / 2

            ce_loss = (self.ce_loss(predict_logits_1, label) + self.ce_loss(predict_logits_2, label)) / 2
            loss = ce_loss + kld_loss * self.pit_weight
            predict_result = torch.argmax(predict_logits_1, dim=1)

            # input_ids = input_ids_2
            # attention_mask = attention_mask_2
            # token_type_ids = token_type_ids_2
            
            # predict_logits = self.predict(input_ids, attention_mask, token_type_ids)
            # loss = self.ce_loss(predict_logits, label)

            # predict_result = torch.argmax(predict_logits, dim=1)

            return loss, predict_result
        else:
            predict_logits = self.predict(input_ids_1, attention_mask_1, token_type_ids_1)
            loss = self.ce_loss(predict_logits, label)
            predict_result = torch.argmax(predict_logits, dim=1)
            
            return loss, predict_result
