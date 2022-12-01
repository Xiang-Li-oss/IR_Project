import os
from typing import Tuple
import torch
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

parser = argparse.ArgumentParser(description='Information Retrieval')
parser.add_argument('--train', action='store_true', help='Whether to train')
parser.add_argument('--test', action='store_true', help='Whether to test')

parser.add_argument('--batch', type=int, default=16, help='Define the batch size')
parser.add_argument('--datetime', type=str, required=True, help='Get Time Stamp')
parser.add_argument('--epoch', type=int, default=50, help='Training epochs')
parser.add_argument('--gpu', type=str, nargs='+', help='Use GPU')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--seed',type=int, default=42, help='Random Seed')
parser.add_argument('--early_stop',type=int, default=10, help='Early Stop Epoch')

parser.add_argument('--bert', type=str, required=True, help='Choose Bert')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout ratio')
parser.add_argument("--save_path", default='./output', type=str,
                        help="The path of result data and models to be saved.")

args = parser.parse_args()

TIMESTAMP = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
SAVE_PATH =  os.path.join(args.save_path, args.model_name,
                                      f'{TIMESTAMP}_{args.batch}_{args.dropout}_{args.learning_rate}')

config_logging(SAVE_PATH)
logging.info('Log is ready!')
logging.info(args)


def build_data_loader() -> Tuple[DataLoader, DataLoader]:
    pass


def build_optimizer() -> Optimizer:
    pass

def build_scheduler() -> _LRScheduler:
    pass

def get_criterion():
    pass

def get_metric():
    pass

def train():
    train_loader, test_loader = build_data_loader()
    model = BertClassifier(args.bert, args.dropout)
    optimizer = build_optimizer()
    scheduler = build_scheduler()
    criterion = get_criterion()

    metric = get_metric()
    best_metric = 0
    metric_not_update = 0

    trainer = Trainer(args, model=model, metric=metric)

    for epoch in range(args.epoch):
        loss = trainer.train(train_loader, optimizer, scheduler, criterion)
        logging.info(f'epoch:{epoch}, loss:{loss}')

        metrics = trainer.eval(test_loader, metric)
        metric = metrics['ndcg_cut_10']
        if metric > best_metric:
            logging.info(f'evaluation result:{metric}')
            trainer.save()
            logging.info(f'state dict saved to {SAVE_PATH}')
        else:
            metric_not_update += 1
            if metric_not_update >= args.early_stop:
                logging.info(f'metric has not updated for {metric_not_update} epochs, stop training')
                break
        


if __name__ == '__main__':
    if args.train:
        train()

   