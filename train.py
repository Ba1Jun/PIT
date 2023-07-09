'''
This script handling the training process.
'''
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import math
import time
import logging
import json
import random
from tqdm import tqdm
from collections import defaultdict

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
# from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_metrics(labels, predict_results):
    # average_method = 'macro'
    average_method = 'binary'
    precision = precision_score(labels, predict_results, average=average_method)
    recall = recall_score(labels, predict_results, average=average_method)
    f1 = f1_score(labels, predict_results, average=average_method)
    accuracy = accuracy_score(labels, predict_results)
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch_i, args):

    model.train()
    total_tr_loss = []
    predict_results = []
    labels = []
    start = time.time()

    for step, batch in enumerate(tqdm(train_dataloader, desc='  -(Train)', leave=False)):
        # forward
        tr_loss, predict_result = model(**batch)

        # backward
        tr_loss.backward()

        # record
        total_tr_loss.append(tr_loss.cpu().detach().item())
        predict_results += predict_result.cpu().detach().tolist()
        labels += batch["label"].cpu().detach().tolist()

        # update parameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        model.zero_grad()
    
    avg_tr_loss = sum(total_tr_loss) / len(total_tr_loss) * 1.0
    metrics_dict = compute_metrics(labels, predict_results)

    logger.info('(Train) loss: {loss: 8.5f}, '\
                'pre: {pre:3.3f} %, recall: {recall:3.3f} %, f1: {f1:3.3f} %, acc: {acc:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(loss=avg_tr_loss, pre=100*metrics_dict['precision'], 
                recall=100*metrics_dict['recall'], f1=100*metrics_dict['f1'], acc=100*metrics_dict['accuracy'], 
                elapse=(time.time()-start)/60))



def eval_epoch(model, eval_dataloader, args, stage='Dev'):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_eval_loss = []
    predict_results = []
    labels = []
    start = time.time()

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f'  -({stage})', leave=False):
            # forward
            eval_loss, predict_result = model(**batch)

            # record
            total_eval_loss.append(eval_loss.cpu().detach().item())
            predict_results += predict_result.cpu().detach().tolist()
            labels += batch["label"].cpu().detach().tolist()
    
    avg_eval_loss = sum(total_eval_loss) / len(total_eval_loss) * 1.0
    metrics_dict = compute_metrics(labels, predict_results)

    logger.info('({stage}) loss: {loss: 8.5f}, pre: {pre:3.3f} %, recall: {recall:3.3f} %, '\
                'f1: {f1:3.3f} %, acc: {acc:3.3f} %, '\
                'elapse: {elapse:3.3f} min'.format(stage=stage, loss=avg_eval_loss, pre=100*metrics_dict['precision'], 
                recall=100*metrics_dict['recall'], f1=100*metrics_dict['f1'], acc=100*metrics_dict['accuracy'], elapse=(time.time()-start)/60))
    
    return metrics_dict


def run(model, train_dataloader, dev_dataloader, test_dataloader, args):
    args.num_train_instances = train_dataloader.dataset.__len__()
    args.num_training_steps =  math.ceil(args.num_train_instances / args.batch_size) * args.epoch
    logger.info("batch size:{}".format(args.batch_size))
    logger.info("total train steps:{}".format(args.num_training_steps))
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(args.warmup_proportion * args.num_training_steps), 
        num_training_steps=args.num_training_steps
    )
    
    best_metric = -1
    best_epoch = 0

    for epoch_i in range(1, args.epoch+1):
        logger.info('[ Epoch{} ]'.format(epoch_i))
        # train 
        train_epoch(model, train_dataloader, optimizer, scheduler, epoch_i, args)
        # dev
        current_metric = eval_epoch(model, dev_dataloader, args, stage='Dev  ')[args.main_metric]
        # test
        _ = eval_epoch(model, test_dataloader, args, stage='Test ')

        if args.save_model_path != "":
            model_name = args.save_model_path + '/model.pt'
            if not os.path.exists(args.save_model_path):
                os.makedirs(args.save_model_path)
            if current_metric >= best_metric:
                best_epoch = epoch_i
                best_metric = current_metric
                checkpoint = {
                    "model_state_dict": model.state_dict()
                }
                torch.save(checkpoint, model_name)
                logger.info('  - [Info] The checkpoint file has been updated.')
    if args.save_model_path != "":
        logger.info(f'Conduct evaluation on test dataset')
        model.load_state_dict(torch.load(model_name)["model_state_dict"])
        model.to(args.device)
        logger.info('reload best checkpoint')
        _ = eval_epoch(model, test_dataloader, args, stage='Test ')
    logger.info(f'Got best test performance on epoch{best_epoch}')



def prepare_dataloaders(args):
    from utils_data import QEDataset as Dataset
    train_dataset = Dataset(args, split="train")
    dev_dataset = Dataset(args, split="dev")
    test_dataset = Dataset(args, split="test")

    logger.info(f"train data size: {train_dataset.__len__()}")
    logger.info(f"dev data size: {dev_dataset.__len__()}")
    logger.info(f"test data size: {test_dataset.__len__()}")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn)

    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn)
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=test_dataset.collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument('--max_len_1', type=int)
    parser.add_argument('--max_len_2', type=int)
    # evaluation
    parser.add_argument('--main_metric', type=str)
    # training
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--warmup_proportion', type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument('--seed', type=int)
    # model
    parser.add_argument('--method', type=str, default='')
    parser.add_argument('--pit_weight', type=float, default=1)
    parser.add_argument('--encoder_type', type=str, default='')
    parser.add_argument('--save_model_path', type=str, default='')


    args = parser.parse_args()

    # set plm path and data file
    from utils_data import prepare_plm_data_path
    prepare_plm_data_path(args)

    logger.info(args)
    
    # set device
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Current random seed: {args.seed}")
    # random seed
    set_seed(args.seed)

    # preparing model
    from models.matching_model_pit import MatchingModel
    model = MatchingModel(args)
    model.to(args.device)

    # loading dataset
    train_dataloader, dev_dataloader, test_dataloader = prepare_dataloaders(args)

    # running
    run(model, train_dataloader, dev_dataloader, test_dataloader, args)


if __name__ == '__main__':
    main()
