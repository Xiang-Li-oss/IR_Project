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
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, token_type_ids, mask):
        output = self.bert(input_ids= input_id, token_type_ids=token_type_ids, attention_mask=mask)
        pooler_output = output['pooler_output'] #Only Embedding for [CLS] is required in classification
        dropout_output = self.dropout(pooler_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

class PretrainedModel(nn.Module):
    def __init__(self, bert, n_labels, feature_layers, dropout):
        super(PretrainedModel, self).__init__()
        self.bert = getBert(bert)
        self.feature_layers = feature_layers
        self.dropout = nn.Dropout(dropout)
        # self.linear = nn.Linear(self.bert.config.hidden_size * feature_layers, n_labels)
 
        # nn.init.normal_(self.linear.weight, std=0.02)
        # nn.init.normal_(self.linear.bias, 0)
        self.linear = nn.Sequential(
            nn.Linear(768, 128),
            nn.Tanh(),
            nn.Linear(128, n_labels)
        )

    def forward(self,input_ids,token_type_ids,attention_mask):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # print(outputs['pooler_output'].shape)
        # print(outputs['last_hidden_state'].shape)
        # print(outputs['hidden_states'].shape)
        # output = torch.cat([outputs['hidden_states'][-i][:, 0] for i in range(1, self.feature_layers+1)], dim=-1)
        output = torch.mean(outputs['last_hidden_state'], 1)
        return self.linear(self.dropout(output))