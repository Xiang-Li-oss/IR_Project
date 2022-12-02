import os
from typing import Tuple
import torch
import torch.nn as nn
import logging
import argparse
import time
from model import BertClassifier
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import tqdm
from trainer import Trainer
from log import config_logging
from model import BertClassifier
from dataset import IRDataset, IR_collate, Metric
from transformers import get_linear_schedule_with_warmup, AdamW



parser = argparse.ArgumentParser(description='Information Retrieval')
parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--test', action='store_true', help='Whether to test')

parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
# parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--epoch', type=int, default=10, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--lr', type=float, default=1e-6, help='learning rate')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')
parser.add_argument('--early_stop',type=int, default=10, help='Early Stop Epoch')

parser.add_argument('--bert', type=str, help='Choose Bert', default='/data/huangziyang/LFY/PTM/bert-base-uncased')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout ratio')
parser.add_argument("--save_path", default='./output', type=str,
                        help="The path of result data and models to be saved.")

parser.add_argument("--model_name", default='try', type=str)
parser.add_argument("--query_source", default='./data/queries.train.sampled.tsv', type=str)
parser.add_argument("--passage_source", default='./data/collection.train.sampled.tsv', type=str)
parser.add_argument("--train_data_path", default='./data/qidpidtriples.train.sampled.tsv', type=str)
parser.add_argument("--dev_data_path", default='./data/msmarco-passagetest2019-43-top1000.tsv', type=str)
parser.add_argument("--test_data_path", default='./data/msmarco-passagetest2020-54-top1000.tsv', type=str)
parser.add_argument("--dev_gold_path", default='./data/2019qrels-pass.txt', type=str)
parser.add_argument("--test_gold_path", default='./data/2020qrels-pass.txt', type=str)


args = parser.parse_args()


TIMESTAMP = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
SAVE_PATH =  os.path.join(args.save_path, args.model_name)

config_logging(SAVE_PATH)
logging.info('Log is ready!')
logging.info(args)


def build_data_loader() -> Tuple[DataLoader, DataLoader]:
    train_dataset = IRDataset(args, is_train=1)
    logging.info(f'train dataset len: {len(train_dataset)}')

    dev_dataset = IRDataset(args, is_train=0)
    logging.info(f"dev dataset len: {len(dev_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, collate_fn=lambda x: IR_collate(x, args, is_train=1))
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch, collate_fn=lambda x: IR_collate(x, args, is_train=0))
    
    return train_loader, dev_loader

    
    

def train():
    train_loader, test_loader = build_data_loader()
    model = BertClassifier(args.bert, args.dropout).cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.epoch
        )
    criterion = nn.BCELoss(reduction='sum')

    metric = Metric(args)
    best_score = 0

    trainer = Trainer(args, model=model)

    for epoch in range(args.epoch):
        trainer.train(train_loader, optimizer, scheduler, criterion, logging)

        score = trainer.eval(test_loader, metric)
        if score > best_score:
            logging.info(f'evaluation result:{score}')
            trainer.save_model(os.path.join(SAVE_PATH, 'ckpt.pt'))
            best_score = score
            logging.info(f'state dict saved to {SAVE_PATH}')
        # else:
        #     metric_not_update += 1
        #     if metric_not_update >= args.early_stop:
        #         logging.info(f'metric has not updated for {metric_not_update} epochs, stop training')
        #         break
            
def test():
    model = BertClassifier(args.bert, args.dropout).cuda()
    logging.info(f"loading trained parameters from {SAVE_PATH}")
    model.load_state_dict(torch.load(os.path.join(SAVE_PATH, 'ckpt.pt')))
    test_dataset = IRDataset(args, is_train=0)
    logging.info(f"test dataset len: {len(test_dataset)}")
    trainer = Trainer(args, model=model)
    metric = Metric(args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, collate_fn=lambda x: IR_collate(x, args, is_train=0))
    score = trainer.eval(test_loader, metric)
    logging.info(f'test result:{score}')


if __name__ == '__main__':
    if args.train:
        train()
    else:
        test()

   