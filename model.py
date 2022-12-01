import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer, AutoConfig

def getBert(bert_name):
    print('load '+ bert_name)
    model_config = AutoConfig.from_pretrained(bert_name)
    bert = AutoModel.from_pretrained(bert_name,config=model_config)
    return bert

def getTokenizer(bert_name):
    print('load '+ bert_name + ' tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(bert_name)
    return tokenizer

class BertClassifier(nn.Module):
    def __init__(self, bert, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = getBert(bert)
        if any(name in bert for name in ['large', 'Ubert']):
            dim = 1024
        elif '1.3B' in bert:
            dim = 2048
        else:
            dim = 768
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, token_type_ids, mask) -> torch.Tensor:
        output = self.bert(input_ids= input_id, token_type_ids=token_type_ids, attention_mask=mask)
        pooler_output = output['pooler_output'] #Only Embedding for [CLS] is required in classification
        dropout_output = self.dropout(pooler_output)
        linear_output: torch.Tensor = self.linear(dropout_output) #[B, 1]
        score = self.sigmoid(linear_output.squeeze(-1))
        return score #[B]

