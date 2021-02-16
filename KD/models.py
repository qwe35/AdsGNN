import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, BertConfig, BertModel

# Pooling strategies
def get_pooled_output(sequence_output, pooled_output, all_hidden_states, layer, attention_mask):
    return pooled_output # [batch, hidden_size]
def get_first_token_output(sequence_output, pooled_output, all_hidden_states, layer, attention_mask):
    # all_hidden_states is: embedding + all layers of hidden output: tuple (size layer_count+1) of [batch_size, max_seq_len, hidden_size], last layer is -1
    return all_hidden_states[layer][:,0,:] # [batch_size, hidden_size]
def get_reduce_mean_output(sequence_output, pooled_output, all_hidden_states, layer, attention_mask):
    target_layer_hidden_states = all_hidden_states[layer] # [batch_size, max_seq_len, hidden_size]
    output = torch.sum(target_layer_hidden_states * attention_mask.unsqueeze(-1), 1) / (torch.sum(attention_mask, -1, keepdim=True) + 1e-10)
    return output # [batch_size, hidden_size]
def get_reduce_max_output(sequence_output, pooled_output, all_hidden_states, layer, attention_mask):
    pass
def get_reduce_mean_max_output(sequence_output, pooled_output, all_hidden_states, layer, attention_mask):
    pass

pooling_method_map = {
    "pooled": get_pooled_output,
    "first_token": get_first_token_output,
    "reduce_mean": get_reduce_mean_output,
    "reduce_max": get_reduce_max_output,
    "reduce_mean_max": get_reduce_mean_max_output
}

class TransformerEncoder(nn.Module):
    def __init__(self, pretrained_weights, dense_sizes, pooling_strategy='reduce_mean', layer=-1, dropout_rate=0.2, use_src_emb=False):
        super().__init__()
        path_lower = pretrained_weights.lower()
        if 'web' in path_lower and 'bert' in path_lower:
            config_path = os.path.join(pretrained_weights, "Web-Bert-V5_config.json")
            model_path = os.path.join(pretrained_weights, "Web-Bert-V5.pt")
            config = BertConfig.from_json_file(config_path)
            self.transformer_model = BertModel(config)
            self.transformer_model.load_state_dict(torch.load(model_path))
            self.pooling_method = pooling_method_map["pooled"]
            print(f"Loaded web bert with config: {config_path}, and model: {model_path}")
        else:
            self.transformer_model = AutoModel.from_pretrained(pretrained_weights)
            self.pooling_method = pooling_method_map[pooling_strategy]
        self.layer = layer
        self.use_src_emb = use_src_emb
        self.dropout = nn.Dropout(dropout_rate)
        dense_layers = []
        input_size = self.transformer_model.config.hidden_size
        if self.use_src_emb:
            input_size += int(dense_sizes.split(',')[-1])
        for dense_size in [int(x) for x in dense_sizes.split(',')]:
            dense_layers.append(nn.Linear(input_size, dense_size))
            input_size = dense_size
        self.dense_laysers = nn.ModuleList(dense_layers)

    def forward(self, input_ids, attention_mask, src_emb):
        sequence_output, pooled_output, all_hidden_states = self.transformer_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        transformer_encoding = self.pooling_method(sequence_output, pooled_output, all_hidden_states, self.layer, attention_mask)
        if self.use_src_emb:
            output = torch.cat([transformer_encoding, src_emb], 1)
        else:
            output = transformer_encoding
        for i in range(len(self.dense_laysers)-1):
            output = F.relu(self.dense_laysers[i](output))
            output = self.dropout(output)
        output = self.dense_laysers[-1](output)
        output = F.normalize(output, dim=-1, p=2)
        return output

class MLP(nn.Module):
    def __init__(self, input_size, dense_sizes, dropout_rate=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        dense_layers = []
        for dense_size in [int(x) for x in dense_sizes.split(',')]:
            dense_layers.append(nn.Linear(input_size, dense_size))
            input_size = dense_size
        self.dense_laysers = nn.ModuleList(dense_layers)

    def forward(self, src_emb):
        output = src_emb
        for i in range(len(self.dense_laysers)-1):
            output = F.relu(self.dense_laysers[i](output))
            output = self.dropout(output)
        output = self.dense_laysers[-1](output)
        output = F.normalize(output, dim=-1, p=2)
        return output