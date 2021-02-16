# -*- coding: UTF-8 -*-
"""
node-level BertGNN model
use transformers bert version https://github.com/huggingface/transformers
Two GnnEncoder
- graphsage
- gat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class BertNodeGnnConfig():
    def __init__(self,
                 bert_model_path_or_name="bert-base-uncased",
                 num_labels=1,
                 gnn_acts="leaky_relu",
                 gnn_fanouts:[list, None]=None,
                 gnn_encoder = 'graphsage',
                 gnn_hidden_size:[list, None]=None,
                 gnn_aggregator="meanpool",
                 gnn_residual:["add","concat"]="add",
                 bert_emb_dim=None,
                 gnn_head_nums=[3],
                 is_freeze_bert=True,
                 inference_neighbor_bert=False,
                 is_use_gnn=True,
                 **kwargs
                 ):
        if gnn_fanouts is None:
            gnn_fanouts = [3]
        self.bert_model_path_or_name = bert_model_path_or_name
        self.num_labels = num_labels
        self.gnn_acts = gnn_acts
        self.gnn_fanouts = gnn_fanouts
        self.gnn_encoder = gnn_encoder
        self.gnn_hidden_size = gnn_hidden_size
        self.gnn_aggregator = gnn_aggregator
        self.gnn_residual = gnn_residual

        """"""
        self.bert_emb_dim = bert_emb_dim
        self.gnn_input_dim = None
        self.gnn_head_nums = gnn_head_nums
        """For bert embedding"""
        self.is_freeze_bert = is_freeze_bert
        self.inference_neighbor_bert = inference_neighbor_bert
        """if use gnn model"""
        self.is_use_gnn = is_use_gnn

class BertNodeGnnModel(nn.Module):
    """
    bert_encoder
    GNN
    pooler
    output
    """
    def __init__(self, global_config:BertNodeGnnConfig):
        super(BertNodeGnnModel, self).__init__()
        self.config = global_config
        self.BERT_init()
        if self.config.is_use_gnn is True:
            self.GNN_init()
        self.predict_init()

    def BERT_init(self):
        self.bert_model = BertModel.from_pretrained(self.config.bert_model_path_or_name)
        self.bert_emb_dim = self.bert_model.config.hidden_size
        self.config.bert_emb_dim = self.bert_emb_dim
        # freeze bert
        if self.config.is_freeze_bert:
            for para in self.bert_model.parameters():
                para.requires_grad = False
        if self.config.is_freeze_bert is False and self.config.inference_neighbor_bert is True:
            print('load freeze checkpiont')
            self.bert_model_freezed = BertModel.from_pretrained(self.config.bert_model_path_or_name)
            # the freeze
            for para in self.bert_model_freezed.parameters():
                para.requires_grad = False

    def GNN_init(self):

        gnn_encoder_type = self.config.gnn_encoder.strip()

        gnn_model_class = {
            "graphsage" : GraphsageEncoder,
            "gat" : GatEncoder
        }.get(gnn_encoder_type, None)

        if gnn_encoder_type is None:
            raise NotImplementedError
        self.gnn_model = gnn_model_class(self.config)

    def predict_init(self):
        if self.config.is_use_gnn is True:
            gnn_output_dim = self.gnn_model.gnn_hidden_dims[-1]
        else:
            gnn_output_dim = self.bert_model.config.hidden_size
        self.num_label = self.config.num_labels
        self.predict_model = nn.Linear(gnn_output_dim*2, self.num_label)


    def update_freezed_bert_parameter(self):
        if self.config.is_freeze_bert is False and self.config.inference_neighbor_bert is True:
            self.bert_model_freezed.load_state_dict(self.bert_model.state_dict())
            # again freeze the paramerters
            for para in self.bert_model_freezed.parameters():
                para.requires_grad = False
        return

    def get_bert_embedding_base(self, bert, input_ids, input_mask, segment_ids):
        outputs = bert(input_ids, input_mask, segment_ids)
        pooled_output = outputs[1]
        return pooled_output

    def get_bert_embedding(self, input_ids, input_mask, segment_ids):
        return self.get_bert_embedding_base(self.bert_model, input_ids, input_mask, segment_ids)

    def get_bert_embedding_freeze(self, input_ids, input_mask, segment_ids):
        return self.get_bert_embedding_base(self.bert_model_freezed, input_ids, input_mask, segment_ids)

    def get_neighbor_embedding(self, input_neighbor):

        if self.config.is_freeze_bert is False and self.config.inference_neighbor_bert is True:
            bert_embed_func = self.get_bert_embedding_freeze
        else:
            bert_embed_func = self.get_bert_embedding

        neigh_cnt = len(input_neighbor) // 3
        neigh_emb_list = []
        for i in range(neigh_cnt):
            neigh_emb_list.append(bert_embed_func(
                input_ids=input_neighbor[i * 3],
                input_mask=input_neighbor[i*3+1],
                segment_ids=input_neighbor[i*3+2]
            ))
        neigh_emb = torch.stack(neigh_emb_list, dim=1)
        return neigh_emb, neigh_cnt # [bs, neigh_cnt, emb_dim]

    def forward(self, q_tensor, k_tensor, q_neighbor, k_neighbor, label_id=None, **kwargs):
        '''
        :param q_tensor: list of tensor: [input_ids, input_mask, segment_ids]
        :param k_tensor: list of tensor: [bs, seq_len]
        :param q_neighbor: list of tensor
        :param k_neighbor: list of tensor
        :param label: [bs]
        :return:
        '''

        """
        one-stage node-level BertGNN forward
        """
        label = label_id
        q_emb = self.get_bert_embedding(input_ids=q_tensor[0], input_mask=q_tensor[1], segment_ids=q_tensor[2])
        k_emb = self.get_bert_embedding(input_ids=k_tensor[0], input_mask=k_tensor[1], segment_ids=k_tensor[2])
        if self.config.is_use_gnn is True:
            q_neigh_emb, q_neigh_cnt = self.get_neighbor_embedding(q_neighbor)
            k_neigh_emb, k_neigh_cnt = self.get_neighbor_embedding(k_neighbor)
            gnn_q_output = self.gnn_model([q_emb, q_neigh_emb], [q_neigh_cnt])
            gnn_k_output = self.gnn_model([k_emb, k_neigh_emb], [k_neigh_cnt])
        else:
            gnn_q_output = q_emb
            gnn_k_output = k_emb
        predict_input = torch.cat([gnn_q_output, gnn_k_output], dim=-1)
        logits = self.predict_model(predict_input)

        outputs = (logits,)
        if label is not None:
            if self.num_label == 1:
                # regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), label.view(-1))
            else:
                # classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_label), label.view(-1))
            outputs = (loss,) + outputs

        return outputs


class GraphsageEncoder(nn.Module):
    def __init__(self, config:BertNodeGnnConfig):
        super(GraphsageEncoder, self).__init__()
        self.config = config
        self.num_layers = len(self.config.gnn_fanouts)
        gnn_hidden_dims = self.config.gnn_hidden_size
        if gnn_hidden_dims is None:
            gnn_hidden_dims = [self.config.bert_emb_dim] * self.num_layers
        self.gnn_hidden_dims = gnn_hidden_dims
        gnn_input_dim = self.config.gnn_input_dim if self.config.gnn_input_dim else self.config.bert_emb_dim
        self.gnn_input_dim = gnn_input_dim
        self.init_layers()

    def init_layers(self):
        gnn_input_dim = self.gnn_input_dim

        for layer_index in range(self.num_layers):
            gnn_output_dim = self.gnn_hidden_dims[layer_index]
            # aggregator
            if self.config.gnn_aggregator == "meanpool":
                setattr(self, "sage_pool_{}".format(layer_index), nn.Linear(gnn_input_dim, gnn_output_dim))
                neighbor_dim = gnn_output_dim
            elif self.config.gnn_aggregator == "lstm":
                setattr(self, "sage_pool_{}".format(layer_index), nn.LSTM(gnn_input_dim, gnn_output_dim, 2))
                neighbor_dim = gnn_output_dim
            else:
                neighbor_dim = gnn_input_dim
            # combiner
            if self.config.gnn_residual == "add":
                setattr(self, "sage_add_{}".format(layer_index), nn.Linear(gnn_input_dim, gnn_output_dim))
                if neighbor_dim != gnn_output_dim:
                    setattr(self, "sage_add_nei_{}".format(layer_index), nn.Linear(neighbor_dim, gnn_output_dim))
            elif self.config.gnn_residual == "concat":
                setattr(self, "sage_concat_{}".format(layer_index), nn.Linear(gnn_input_dim+neighbor_dim, gnn_output_dim))
            else:
                raise NotImplementedError("invalid gnn residual : {}".format(self.config.gnn_residual))
            # update input dim for next layer
            gnn_input_dim = gnn_output_dim

    def forward(self, node_emb, fanouts):
        """
        :param node_emb: list
        :param fanouts: list like [3,3], default [3]
        :return: [bs, out_dim]
        """
        if self.num_layers == 0:
            return node_emb[0]

        for layer_index in range(self.num_layers):
            hidden = []
            for hop in range(self.num_layers - layer_index):
                neigh_emb = torch.reshape(node_emb[hop+1], [-1, fanouts[hop], node_emb[hop+1].shape[-1]]) # [bs, neigh_cnt, feat_dim]
                if self.config.gnn_aggregator == "meanpool":
                    neigh_emb = getattr(self, "sage_pool_{}".format(layer_index))(neigh_emb)
                    neigh_emb = torch.mean(neigh_emb, dim=1)
                elif self.config.gnn_aggregator == "mean":
                    neigh_emb = torch.mean(neigh_emb, dim=1)
                elif self.config.gnn_aggregator == "lstm":
                    out = getattr(self, "sage_pool_{}".format(layer_index))(neigh_emb)
                    neigh_emb = out[0]
                    neigh_emb = torch.mean(neigh_emb, dim=1)
                if self.config.gnn_residual == "add":
                    node = getattr(self, "sage_add_{}".format(layer_index))(node_emb[hop])
                    if neigh_emb.shape[-1] != node.shape[-1]:
                        neigh_emb = getattr(self, "sage_add_nei_{}".format(layer_index))(neigh_emb)
                    hidden.append(node + neigh_emb)
                elif self.config.gnn_residual == "concat":
                    seq = torch.cat([node_emb[hop], neigh_emb], dim=-1)
                    seq = getattr(self, "sage_concat_{}".format(layer_index))(seq)
                    hidden.append(seq)
            # end this layer
            node_emb = hidden
        # [bs, out_dim]
        return node_emb[0]

class AttnHead(nn.Module):
    """
    attn head for GNN aggregation part
    """
    def __init__(self, input_size, out_size, headindex=None, act_fn=F.relu, residual=False):
        super(AttnHead, self).__init__()
        self.head_name = headindex
        self.linear_layer_fts = nn.Linear(in_features=input_size, out_features=out_size, bias=False)
        self.linear_layer_f_1 = nn.Linear(in_features=out_size, out_features=1, bias=True)
        self.linear_layer_f_2 = nn.Linear(in_features=out_size, out_features=1, bias=True)
        self.ACT_FN = act_fn
        self.residual = residual
        self.bias = nn.Parameter(torch.Tensor(1))
        if residual:
            self.linear_layer_residual = nn.Linear(in_features=input_size, out_features=out_size, bias=True)

        self.weight_init()

    def weight_init(self):
        nn.init.zeros_(self.bias)


    def forward(self, seq): # [bs, 3, 1024]
        seq_fts = self.linear_layer_fts(seq) # [bs, 3, 1024]
        f_1 = self.linear_layer_f_1(seq) # [bs, 3, 1]
        f_2 = self.linear_layer_f_2(seq) # [bs, 3, 1]

        logits = f_1 + f_2.permute(0, 2, 1) # [bs, 3, 3]
        coefs = F.softmax(F.leaky_relu(logits)) # [bs, 3, 3]
        vals = torch.matmul(coefs, seq_fts) # [bs, 3, 1024]

        ret = vals + self.bias

        # residual connection
        if self.residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.linear_layer_residual(seq)
            else:
                ret = ret + seq
        # return
        return self.ACT_FN(ret)

class AttnHeadLayer(nn.Module):
    def __init__(self, input_dim, output_dim, gnn_residual="add", activation=F.leaky_relu):
        super(AttnHeadLayer, self).__init__()
        self.conv = nn.Linear(input_dim, output_dim, bias=False)
        self.conv2_1 = nn.Linear(output_dim, 1)
        self.conv2_2 = nn.Linear(output_dim, 1)
        self.output_bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.float, requires_grad=True))
        self.res = None
        self.gnn_residual = gnn_residual
        if gnn_residual == "add":
            if input_dim != output_dim:
                self.res = nn.Linear(input_dim, output_dim)
        elif gnn_residual == "concat":
            self.res = nn.Linear(input_dim+output_dim, output_dim)
        self.activation = activation

    def forward(self, seq):
        seq_fts = self.conv(seq)
        f_1 = self.conv2_1(seq_fts)
        f_2 = self.conv2_2(seq_fts)
        logits = f_1 + torch.transpose(f_2, 1, 2)
        coefs = F.softmax(F.leaky_relu(logits), -1)
        vals = torch.matmul(coefs, seq_fts)
        ret = vals + self.output_bias
        if self.gnn_residual == "add":
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + self.res(seq)
            else:
                ret = ret + seq
        elif self.gnn_residual == "concat":
            ret = torch.cat([ret, seq], dim=-1)
            ret = self.res(ret)
        ret = self.activation(ret)
        return ret


class GatEncoder(nn.Module):
    def __init__(self, config:BertNodeGnnModel):
        super(GatEncoder, self).__init__()
        self.config = config
        self.gnn_head_nums = self.config.gnn_head_nums
        self.num_layers = len(self.config.gnn_fanouts)
        gnn_hidden_dims = self.config.gnn_hidden_size
        if gnn_hidden_dims is None:
            gnn_hidden_dims = [self.config.bert_emb_dim] * self.num_layers
        self.gnn_hidden_dims = gnn_hidden_dims
        # activation for each layer
        self.gnn_activations = []
        for act in self.config.gnn_acts.split(','):
            if act == "relu":
                activation = F.relu
            elif act == "leaky_relu":
                activation = F.leaky_relu
            else:
                raise NotImplementedError
            self.gnn_activations.append(activation)

        self.attn_head_layers = []
        input_dim = self.config.bert_emb_dim
        for layer_idx, head_num in enumerate(self.gnn_head_nums):
            output_dim = self.gnn_hidden_dims[layer_idx]
            for head_idx in range(head_num):

                setattr(self, "attn_layer_{}_head_{}".format(layer_idx, head_idx),
                        AttnHeadLayer(input_dim, output_dim, gnn_residual=self.config.gnn_residual, activation=self.gnn_activations[layer_idx]))
            input_dim = output_dim * head_num


        self.print = False

    def forward(self, samples, fanouts):
        if len(fanouts) == 0:
            return samples[0]
        neigh_emb = torch.reshape(samples[1], [-1, fanouts[0], samples[0].shape[-1]])
        seq = torch.cat([torch.unsqueeze(samples[0], 1), neigh_emb], dim=1)


        for layer_idx, head_num in enumerate(self.gnn_head_nums):
            hidden = []
            for head_idx in range(head_num):
                hidden_val = getattr(self, "attn_layer_{}_head_{}".format(layer_idx, head_idx))(seq=seq)
                hidden.append(hidden_val)
            seq = torch.cat(hidden, dim=-1)

        att_output = torch.stack(hidden, 1)     # [bs, head_num, neighbor_num, gat_output_dim]
        att_output = att_output[:, :, 0, :]     # [bs, head_num, gat_output_dim]
        att_output = torch.mean(att_output, 1)  # [bs, gat_output_dim]
        return att_output



def gnn_demo():
    config = BertNodeGnnConfig(
        gnn_encoder='graphsage',
        num_labels=2,
        gnn_fanouts=[3],
        gnn_aggregator='lstm',
        is_freeze_bert=False,
        inference_neighbor_bert=True,
        is_use_gnn=False
    )
    # config = BertNodeGnnConfig(
    #     gnn_encoder='gat',
    #     num_labels=2,
    #     gnn_fanouts=[3],
    #     gnn_aggregator='concat',
    #     is_freeze_bert=False,
    #     inference_neighbor_bert=True,
    # )
    model = BertNodeGnnModel(config)

    txt = "I am Happy today"
    # after BertTokenizer
    txt_token = {'input_ids': [101, 1045, 2572, 3407, 2651, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1]}
    batch_size = 32
    txt_tensor = [
        torch.LongTensor([[101, 1045, 2572, 3407, 2651, 102]] * batch_size),
        torch.LongTensor([[0, 0, 0, 0, 0, 0]] * batch_size),
        torch.LongTensor([[1, 1, 1, 1, 1, 1]] * batch_size)
    ]
    q_tensor = txt_tensor
    k_tensor = txt_tensor
    q_neighbor = txt_tensor + txt_tensor + txt_tensor + txt_tensor
    k_neighbor = txt_tensor + txt_tensor + txt_tensor + txt_tensor
    label = torch.LongTensor([1]*batch_size)

    logits = model(
        q_tensor, k_tensor, q_neighbor, k_neighbor
    )[0]
    print(logits)
    loss = model(
        q_tensor, k_tensor, q_neighbor, k_neighbor, label
    )[0]
    print(loss)
if __name__ == '__main__':
    gnn_demo()

