import torch
from model import BertClassifier
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class Trainer():

    def __init__(self, args, model) -> None:
        self.model:BertClassifier = model
        self.threshold = 0.5
        

    def train(self, train_loader:DataLoader, optimizer:Optimizer, scheduler, criterion, logging):

        model = self.model
        model.train()
        averge_step = len(train_loader) // 8
        step = 0
        total_loss = 0

        for i, input in enumerate(tqdm(train_loader)):
            input_ids:torch.Tensor = input['input_ids']
            attention_mask:torch.Tensor = input['attention_mask']
            label: torch.Tensor = input['label']

            score = model(input_ids, attention_mask)
            loss = criterion(score, label.float())

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss
            step += 1
            if i % averge_step == 0:
                logging.info("Training Loss [{0:.5f}]".format(total_loss/step))
                total_loss, step = 0, 0


    def eval(self, eval_loader, metric):
        model = self.model
        model.eval()
        all_preds = []

        for step, input in enumerate(tqdm(eval_loader)):
            input_ids:torch.Tensor = input['input_ids']
            attention_mask:torch.Tensor = input['attention_mask']

            score = model(input_ids, attention_mask)
            # 这里存它的相反数，方便argsort
            pred = [(input['qid'][i], input['pid'][i], -score[i].item()) for i in range(len(input['qid']))]
            all_preds.extend(pred)
        
        score = metric.get_ndcg(all_preds)
        return score

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        

