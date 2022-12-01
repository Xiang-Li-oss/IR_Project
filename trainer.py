import torch
from model import BertClassifier
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import tqdm


class Trainer():

    def __init__(self,args, model, metric) -> None:
        self.model:BertClassifier = model
        self.cuda = args.cuda
        self.threshold = args.threshold
        

    def train(self, train_loader:DataLoader, optimizer:Optimizer, scheduler, criterion):
        tqdm_loader = tqdm.tqdm(train_loader)
        model = self.model
        model.train()
        total_loss = 0

        for step, input in enumerate(tqdm_loader):
            input_ids:torch.Tensor = input['input_ids']
            token_type_ids:torch.Tensor = input['token_type_ids']
            attention_mask:torch.Tensor = input['attention_mask']
            label: torch.Tensor = input['label']

            if self.cuda:
                model = model.cuda()
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                label:torch.Tensor = label.cuda()


            score = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(score, label)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss
            # pred = torch.where(score > 0.5,
            #                     torch.tensor([1]).to(score.device),
            #                     torch.tensor([0]).to(score.device))

        return total_loss / len(tqdm_loader), 


    def eval(self, eval_loader, metric):
        tqdm_loader = tqdm(eval_loader)
        model = self.model
        model.eval()
        all_preds, all_labels = [], []

        for step, input in enumerate(tqdm_loader):
            input_ids:torch.Tensor = input['input_ids']
            token_type_ids:torch.Tensor = input['token_type_ids']
            attention_mask:torch.Tensor = input['attention_mask']
            label: torch.Tensor = input['label']

            if self.cuda:
                model = model.cuda()
                input_ids = input_ids.cuda()
                token_type_ids = token_type_ids.cuda()
                attention_mask = attention_mask.cuda()
                label:torch.Tensor = label.cuda()


            score = model(input_ids, token_type_ids, attention_mask)

            pred = torch.where(score > self.threshold,
                                torch.tensor([1]).to(score.device),
                                torch.tensor([0]).to(score.device))

            all_preds.append(pred)
            all_labels.append(label)

        metrics = metric(all_preds, all_labels)
        return metrics

