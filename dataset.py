import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import *
import subprocess

class IRDataset(Dataset):
    def __init__(self, args, is_train):
        super(IRDataset, self).__init__()
        self.is_train = is_train
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert)
        self.data = []

        query_source = load_source(args.query_source)
        passage_source = load_source(args.passage_source)
        
        if is_train:
            train_triples = load_tsv(args.train_data_path)[:100]
            for i in tqdm(range(len(train_triples))):
                question = query_source[train_triples[i][0]]
                pos_passage = passage_source[train_triples[i][1]]
                neg_passage = passage_source[train_triples[i][2]]
                question_ids = self.tokenizer.encode(question)
                pos_passage_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pos_passage))
                neg_passage_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(neg_passage))
                pos_input_ids = question_ids + pos_passage_ids + [self.tokenizer.sep_token_id]
                neg_input_ids = question_ids + neg_passage_ids + [self.tokenizer.sep_token_id]
                self.data.append({"input_ids": pos_input_ids, "qid": train_triples[i][0], "pid": train_triples[i][1], "label":1})
                self.data.append({"input_ids": neg_input_ids, "qid": train_triples[i][0], "pid": train_triples[i][2], "label":0})
        else:
            if args.train:
                dev_data = load_tsv(args.dev_data_path)
            else:
                dev_data = load_tsv(args.test_data_path)
            for i in tqdm(range(len(dev_data))):
                question = dev_data[i][2]
                passage = dev_data[i][3]
                question_id = dev_data[i][0]
                passage_id = dev_data[i][1]
                question_ids = self.tokenizer.encode(question)
                passage_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(passage))
                input_ids = question_ids + passage_ids + [self.tokenizer.sep_token_id]
                self.data.append({"input_ids": input_ids, "qid": question_id, "pid": passage_id})
            
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]
        


def IR_collate(data, args, is_train):
    tokenizer = AutoTokenizer.from_pretrained(args.bert)
    max_input_length = max([len(item['input_ids']) for item in data])
    max_input_length = min(max_input_length, 512)
    input_ids = []
    labels = []
    qids = []
    pids = []
    for i in range(len(data)):
        if len(data[i]['input_ids']) > max_input_length:
            input_ids.append(data[i]['input_ids'][:max_input_length])
        else:
            input_ids.append(data[i]['input_ids'] + [tokenizer.pad_token_id] * (max_input_length - len(data[i]['input_ids'])))
        if is_train:
            labels.append(data[i]['label'])
        qids.append(data[i]['qid'])
        pids.append(data[i]['pid'])

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.where(input_ids==tokenizer.pad_token_id, 0, 1)
    labels = torch.tensor(labels)
    if is_train:
        return {"input_ids": input_ids.cuda(), "attention_mask":attention_mask.cuda(), "label":labels.cuda()}
    else:
        return {"input_ids": input_ids.cuda(), "attention_mask": attention_mask.cuda(), "qid": qids, "pid":pids}
    
    
    
    
class Metric:
    def __init__(self, args):
        self.args = args
    
    def get_ndcg(self, preds):
        
        dtype = [('qid', int), ('pid', int), ('score', float)]
        preds = np.array(preds, dtype=dtype)
        preds = preds[np.argsort(preds, order=['qid', 'score'])]

        outputs = []
        current_qid = 0
        rank = 0
        for line in preds:  # 生成rank
            if current_qid != line[0]:
                rank = 1
                current_qid = line[0]
            outputs.append((line[0], line[1], rank, -line[2]))
            rank += 1
        
        # 调用trec_eval程序， 用subprocess读取cmd输出，提取其中的necg@10分数
        if self.args.train:  # dev 
            with open('dev_result.txt', 'w') as f:
                for line in outputs:
                    f.write(f"{line[0]} Q0 {line[1]} {line[2]} {line[3]} CIPZHAO\n")
                f.close()
            p = subprocess.Popen('./trec_eval -m ndcg_cut ./data/2019qrels-pass.txt ./dev_result.txt', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        else:          # test
            with open('test_result.txt', 'w') as f:
                for line in outputs:
                    f.write(f"{line[0]} Q0 {line[1]} {line[2]} {line[3]} CIPZHAO\n")
                f.close()
            p = subprocess.Popen('./trec_eval -m ndcg_cut ./data/2020qrels-pass.txt ./test_result.txt', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        outputs = []
        # 读取cmd的输出
        for line in p.stdout.readlines():
            outputs.append(line.decode())
        # 进行一些字符串处理
        return float(outputs[1].strip().replace('ndcg_cut_10           \tall\t',''))