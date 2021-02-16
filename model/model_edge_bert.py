# -*- coding: UTF-8 -*-
"""
Edge Bert
- Bert encoder for node pair
- Edge-level GNN model
reference transformers BERT version 3.0.2 https://github.com/huggingface/transformers
"""
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel, BertConfig

from .model_token_net import BertModel as BertModelWithTokenNet


# att_head(seq, out_size,headindex, activation=tf.nn.leaky_relu, residual=False):

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


class MultiAttnHead(nn.Module):
    """
    Multi head attention version
    """
    def __init__(self, head_nums, input_size, out_size, act_fn, residual=False):
        super(MultiAttnHead, self).__init__()
        self.head_nums = head_nums
        self.attn_heads = []
        for i in range(head_nums):
            self.attn_heads.append(AttnHead(input_size=input_size, out_size=out_size, headindex=i, act_fn=act_fn, residual=residual))

    def forward(self, seq):
        hidden = []
        for i in range(self.head_nums):
            hidden.append(self.attn_heads[i](seq))
        return hidden


class EdgeBertBlock(nn.Module):
    """
    GNN for edge bert
    Input:
        embedding for nodes and edges
    Ouput:
        network representation
    Args:
        input_hidden_size: bert hidden size
        edgebert_hidden_size: hidden size
        gat_acts: [elu, relu, leaky_relu]
        aggregator: [mean, max, attention]
    """
    def __init__(self, use_node=True, input_hidden_size=768, edgebert_hidden_size=None, gat_acts='relu',
                 aggregator='mean', head_nums=1, use_residual=False):
        super(EdgeBertBlock, self).__init__()

        self.use_node = use_node

        # modify activation function
        if gat_acts == "elu":
            ACT_FN = F.elu
        elif gat_acts == "relu":
            ACT_FN = F.relu
        elif gat_acts == "leaky_relu":
            ACT_FN = F.leaky_relu
        else:
            raise NotImplementedError
        self.GAT_ACT = ACT_FN

        # define the linear layer
        if use_node:
            net_input_dim = input_hidden_size * 2
        else:
            net_input_dim = input_hidden_size
        self.linear_layer = nn.Linear(
            in_features=net_input_dim,
            out_features=edgebert_hidden_size,
            bias=True
        )

        # define the hidden layer
        self.hidden_layer =  nn.Linear(
            in_features=edgebert_hidden_size,
            out_features=edgebert_hidden_size,
            bias=True
        )

        self.aggregator = aggregator

        self.head_nums = head_nums
        self.edgebert_hidden_size = edgebert_hidden_size
        if self.aggregator == 'attention':
            self.multiattn = MultiAttnHead(head_nums=head_nums, input_size=edgebert_hidden_size, out_size=edgebert_hidden_size,
                                           act_fn=ACT_FN, residual=use_residual)

        # init weight
        self.weights_init()

    def weights_init(self):
        nn.init.xavier_uniform_(self.linear_layer.weight)
        # nn.init.constant_(self.linear_layer.bias, 0)
        nn.init.xavier_uniform_(self.hidden_layer.weight)

    def forward(self, q, qk1, qk2, qk3):
        if self.use_node:
            node_feats = q.unsqueeze(1) # [bs, 768] to [bs, 1, 768]
            neighbor_feats = torch.stack([qk1, qk2, qk3], dim=1) # [bs, 3, 768]
            node_feats = node_feats.repeat([1, 3, 1]) # [bs, 1, 768] to [bs, 3, 768]
            edge_feature = torch.cat([node_feats, neighbor_feats], dim=-1) # [bs, 3, 768*2]
        else:
            edge_feature = torch.stack([qk1, qk2, qk3], dim=1) #

        # the first linear layer to edge_feature_hidden_size
        net = self.linear_layer(edge_feature) # [bs, 3, 1024]
        # activation
        net = self.GAT_ACT(net)
        # the aggregation part
        if self.aggregator == 'mean':
            net = torch.mean(net, dim=-2) # [bs, 1024]
        elif self.aggregator == 'max':
            net = torch.max(net, dim=-2)[0] # for torch.max return [tensor, LongTensor]
        elif self.aggregator == 'attention':
            seq = net.reshape([-1, 3, net.shape[-1]]) # [bs, 3, 1024]
            # use multi-head attention
            out = self.multiattn(seq) # list of [bs, 3, 1024]
            # mean pooling
            out = sum(out) / self.head_nums # [bs, 3, 1024]
            # then slice to [bs, 1024]
            out = out[:,0,:]
            return out.reshape([-1, self.edgebert_hidden_size]) # [bs*3, 1024]

        # other cases
        net = torch.squeeze(net) # [bs, 256]
        hidden = net
        hidden = hidden.reshape([-1, self.edgebert_hidden_size]) # [bs, 256]
        net = self.hidden_layer(hidden)
        net = self.GAT_ACT(net)
        return net # [bs, 256]

class ModelQKEdgeBertConfig(object):
    """
    config file for ModelQKEdgeBert
    """
    def __init__(self,
                 edgebert_hidden_size=1024,
                 use_kkqq=True,
                 use_qk=True,
                 use_node=True,
                 gat_acts='relu',
                 aggregator='attention',
                 head_nums=8,
                 comb_loss=True,
                 use_residual=True,
                 num_labels=1,
                 bert_model_path_or_name="bert-base-uncased",
                 is_freeze_bert=True,
                 inference_neighbor_bert=False,
                 gnn_token_embedding_path=None,
                 is_freeze_gnn_token_embedding=True,
                 ):
        """
        :param edgebert_hidden_size: size for hiddene state of edgebert model
        :param use_kkqq:
        :param use_qk:
        :param use_node:
        :param gat_acts:
        :param aggregator:
        :param head_nums:
        :param comb_loss:
        :param use_residual:
        :param num_labels:
        :param bert_model_path_or_name:
        :param is_freeze_bert: whether to freeze the bert part model parameter
        :param inference_neighbor_bert: if donnot freeze bert, only grad the qk edge pair, inference mode for neighbors
        """
        self.edgebert_hidden_size=edgebert_hidden_size
        self.use_kkqq=use_kkqq
        self.use_qk=use_qk
        self.use_node=use_node
        self.gat_acts=gat_acts
        self.aggregator=aggregator
        self.head_nums=head_nums
        self.comb_loss=comb_loss
        self.use_residual=use_residual
        self.num_labels=num_labels
        self.bert_model_path_or_name=bert_model_path_or_name
        self.is_freeze_bert=is_freeze_bert
        self.inference_neighbor_bert=inference_neighbor_bert
        self.gnn_token_embedding_path = gnn_token_embedding_path
        self.is_freeze_gnn_token_embedding = is_freeze_gnn_token_embedding

class ModelQKEdgeBert(nn.Module):
    def __init__(self, args=None):
        """
        Contain following contents:
        - bert model (need to freeze parameter)
            - bert_model_path_or_name
        - edgebert model

        """
        super(ModelQKEdgeBert, self).__init__()
        # bert config
        # neighbor inference setting
        self.is_freeze_bert = args.is_freeze_bert
        self.inference_neighbor_bert = args.inference_neighbor_bert

        self.bert_model_qk = BertModel.from_pretrained(args.bert_model_path_or_name)
        self.bert_hidden_size = self.bert_model_qk.config.hidden_size
        if args.is_freeze_bert:
            for para in self.bert_model_qk.parameters():
                para.requires_grad = False
        if self.is_freeze_bert is False and self.inference_neighbor_bert is True:
            print('load freeze checkpiont')
            self.bert_model_qk_freezed = BertModel.from_pretrained(args.bert_model_path_or_name)
            for para in self.bert_model_qk_freezed.parameters():
                para.requires_grad = False

        # dropout layer
        self.output_layer_qk_dropout = nn.Dropout(p=0.1) # for torch, p – probability of an element to be zeroed. Default: 0.5
        # graph part
        self.edgebert_hidden_size = args.edgebert_hidden_size
        self.use_kkqq = args.use_kkqq
        self.use_qk = args.use_qk
        self.edge_bert_layer = EdgeBertBlock(
            use_node=args.use_node, input_hidden_size=self.bert_hidden_size,
            edgebert_hidden_size=self.edgebert_hidden_size, gat_acts=args.gat_acts,
            aggregator=args.aggregator, head_nums=args.head_nums, use_residual=args.use_residual)
        # output layer
        self.output_layer_ori = nn.Linear(in_features=self.bert_hidden_size, out_features=args.num_labels, bias=True)
        # output for edgebert
        if self.use_kkqq and self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.edgebert_hidden_size*4, out_features=args.num_labels, bias=False)
        elif not self.use_kkqq and self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.edgebert_hidden_size*2, out_features=args.num_labels, bias=False)
        elif self.use_kkqq and not self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.edgebert_hidden_size*2, out_features=args.num_labels, bias=False)
        else:
            raise ValueError("use_qk and use_kkqq can not be False in the same time")
        # loss part
        self.comb_loss = args.comb_loss
        # label part
        self.num_labels = args.num_labels


    def get_qk_bert_embedding(self, input_ids, input_mask, segment_ids):
        outputs = self.bert_model_qk(input_ids, input_mask, segment_ids)
        pooled_output = outputs[1]
        return pooled_output

    def get_qk_freezed_bert_embedding(self, input_ids, input_mask, segment_ids):
        outputs = self.bert_model_qk_freezed(input_ids, input_mask, segment_ids)
        pooled_output = outputs[1]
        return pooled_output

    def update_freezed_bert_parameter(self):
        if self.is_freeze_bert is False and self.inference_neighbor_bert is True:
            self.bert_model_qk_freezed.load_state_dict(self.bert_model_qk.state_dict())
            # again freeze the paramerters
            for para in self.bert_model_qk_freezed.parameters():
                para.requires_grad = False
        return


    def forward(self, input_ids, input_mask, segment_ids, label_id=None, input_neighbor=None,namelist=None, **kwargs):
        # current qk embedding
        output_layer_qk = self.get_qk_bert_embedding(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids) # [bs, 768]
        # for neighbor edges
        output_layers = []
        if self.is_freeze_bert is False and self.inference_neighbor_bert is True:
            # if do not freeze bert and use the neighbor inference mode
            for i in range(len(namelist)):
                output_layers.append(self.get_qk_freezed_bert_embedding(
                    input_ids=input_neighbor[i * 3], input_mask=input_neighbor[i * 3 + 1],
                    segment_ids=input_neighbor[i * 3 + 2]))
        else:
            for i in range(len(namelist)):
                output_layers.append(self.get_qk_bert_embedding(
                    input_ids=input_neighbor[i*3], input_mask=input_neighbor[i*3+1], segment_ids=input_neighbor[i*3+2]))
        # dropout part
        output_layer_qk = self.output_layer_qk_dropout(output_layer_qk)
        for i in range(len(namelist)):
            output_layers[i] = self.output_layer_qk_dropout(output_layers[i])
        # graph part
        # do edgebert for qks, kqs, qqs, kks
        if self.use_kkqq and self.use_qk:
            qself_out, kself_out, qk1_out, qk2_out, qk3_out, kq1_out, kq2_out, kq3_out, kk1_out, kk2_out, kk3_out, qq1_out, qq2_out, qq3_out = output_layers
            qks_emb = self.edge_bert_layer(qself_out,qk1_out,qk2_out,qk3_out)
            kqs_emb = self.edge_bert_layer(kself_out,kq1_out,kq2_out,kq3_out)
            qqs_emb = self.edge_bert_layer(qself_out,qq1_out,qq2_out,qq3_out)
            kks_emb = self.edge_bert_layer(kself_out,kk1_out,kk2_out,kk3_out)
            output_layer = torch.cat([output_layer_qk,qks_emb,kqs_emb,qqs_emb,kks_emb], dim=-1)
        elif not self.use_kkqq and self.use_qk:
            qself_out, kself_out, qk1_out, qk2_out, qk3_out, kq1_out, kq2_out, kq3_out = output_layers
            qks_emb = self.edge_bert_layer(qself_out, qk1_out, qk2_out, qk3_out)
            kqs_emb = self.edge_bert_layer(kself_out, kq1_out, kq2_out, kq3_out)
            output_layer = torch.cat([output_layer_qk, qks_emb, kqs_emb], dim=-1)
        elif self.use_kkqq and not self.use_qk:
            qself_out, kself_out, kk1_out, kk2_out, kk3_out, qq1_out, qq2_out, qq3_out = output_layers
            qqs_emb = self.edge_bert_layer(qself_out,qq1_out,qq2_out,qq3_out)
            kks_emb = self.edge_bert_layer(kself_out,kk1_out,kk2_out,kk3_out)
            output_layer = torch.cat([output_layer_qk, qqs_emb, kks_emb], dim=-1)
        else:
            raise ValueError
        # probilities setting
        _logits_ori = self.output_layer_ori(output_layer_qk)
        _logits_new = self.output_layer_edgebert(output_layer)

        if self.comb_loss:
            _logits = 0.2*_logits_ori + 0.8*_logits_new
        else:
            _logits = _logits_new

        # output part use transformers-style for comparison
        # original transformers use argmax for classification
        # for qk classification inference, see more in evaluate function
        # preds = F.softmax(_logits, dim=-1) # [bs, 2]
        # preds = preds[:, 1] # target score for roc_auc

        # update loss
        outputs = (_logits,)

        if label_id is not None:
            if self.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(_logits.view(-1), label_id.view(-1))
            else:
                # classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(_logits.view(-1, self.num_labels), label_id.view(-1))
            outputs = (loss,) + outputs
        return outputs


class ModelQKEdgeBertWithTokenNet(nn.Module):
    def __init__(self, args=None):
        """
        Contain following contents:
        - bert model (need to freeze parameter)
            - bert_model_path_or_name
        - edgebert model

        Use BertModelWithTokenNet as the Bert Encoder

        """
        super(ModelQKEdgeBertWithTokenNet, self).__init__()
        # bert config
        # neighbor inference setting
        self.is_freeze_bert = args.is_freeze_bert
        self.inference_neighbor_bert = args.inference_neighbor_bert

        if args.gnn_token_embedding_path is None or os.path.exists(args.gnn_token_embedding_path) is False:
            raise ValueError("Unvalid gnn_token_embedding_path : {}".format(args.gnn_token_embedding_path))

        self.bert_model_qk = BertModelWithTokenNet.from_pretrained(args.bert_model_path_or_name)
        self.bert_model_qk.update_token_net_embedding(args.gnn_token_embedding_path, freeze_parameter=args.is_freeze_gnn_token_embedding)
        self.bert_hidden_size = self.bert_model_qk.config.hidden_size
        if args.is_freeze_bert:
            for para in self.bert_model_qk.parameters():
                para.requires_grad = False
        if self.is_freeze_bert is False and self.inference_neighbor_bert is True:
            print('load freeze checkpiont')
            self.bert_model_qk_freezed = BertModelWithTokenNet.from_pretrained(args.bert_model_path_or_name)
            self.bert_model_qk_freezed.update_token_net_embedding(args.gnn_token_embedding_path, freeze_parameter=args.is_freeze_gnn_token_embedding)
            for para in self.bert_model_qk_freezed.parameters():
                para.requires_grad = False

        # dropout layer
        self.output_layer_qk_dropout = nn.Dropout(p=0.1) # for torch, p – probability of an element to be zeroed. Default: 0.5
        # graph part
        self.edgebert_hidden_size = args.edgebert_hidden_size
        self.use_kkqq = args.use_kkqq
        self.use_qk = args.use_qk
        self.edge_bert_layer = EdgeBertBlock(
            use_node=args.use_node, input_hidden_size=self.bert_hidden_size,
            edgebert_hidden_size=self.edgebert_hidden_size, gat_acts=args.gat_acts,
            aggregator=args.aggregator, head_nums=args.head_nums, use_residual=args.use_residual)
        # output layer
        self.output_layer_ori = nn.Linear(in_features=self.bert_hidden_size, out_features=args.num_labels, bias=True)
        # output for edgebert
        if self.use_kkqq and self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.edgebert_hidden_size*4, out_features=args.num_labels, bias=False)
        elif not self.use_kkqq and self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.edgebert_hidden_size*2, out_features=args.num_labels, bias=False)
        elif self.use_kkqq and not self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.edgebert_hidden_size*2, out_features=args.num_labels, bias=False)
        else:
            raise ValueError("use_qk and use_kkqq can not be False in the same time")
        # loss part
        self.comb_loss = args.comb_loss
        # label part
        self.num_labels = args.num_labels


    def get_qk_bert_embedding(self, input_ids, input_mask, segment_ids):
        outputs = self.bert_model_qk(input_ids, input_mask, segment_ids)
        pooled_output = outputs[1]
        return pooled_output

    def get_qk_freezed_bert_embedding(self, input_ids, input_mask, segment_ids):
        outputs = self.bert_model_qk_freezed(input_ids, input_mask, segment_ids)
        pooled_output = outputs[1]
        return pooled_output

    def update_freezed_bert_parameter(self):
        if self.is_freeze_bert is False and self.inference_neighbor_bert is True:
            self.bert_model_qk_freezed.load_state_dict(self.bert_model_qk.state_dict())
            # again freeze the paramerters
            for para in self.bert_model_qk_freezed.parameters():
                para.requires_grad = False
        return


    def forward(self, input_ids, input_mask, segment_ids, label_id=None, input_neighbor=None,namelist=None, **kwargs):
        # current qk embedding
        output_layer_qk = self.get_qk_bert_embedding(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids) # [bs, 768]
        # for neighbor edges
        output_layers = []
        if self.is_freeze_bert is False and self.inference_neighbor_bert is True:
            # if do not freeze bert and use the neighbor inference mode
            for i in range(len(namelist)):
                output_layers.append(self.get_qk_freezed_bert_embedding(
                    input_ids=input_neighbor[i * 3], input_mask=input_neighbor[i * 3 + 1],
                    segment_ids=input_neighbor[i * 3 + 2]))
        else:
            for i in range(len(namelist)):
                output_layers.append(self.get_qk_bert_embedding(
                    input_ids=input_neighbor[i*3], input_mask=input_neighbor[i*3+1], segment_ids=input_neighbor[i*3+2]))
        # dropout part
        output_layer_qk = self.output_layer_qk_dropout(output_layer_qk)
        for i in range(len(namelist)):
            output_layers[i] = self.output_layer_qk_dropout(output_layers[i])
        # graph part
        # do edgebert for qks, kqs, qqs, kks
        if self.use_kkqq and self.use_qk:
            qself_out, kself_out, qk1_out, qk2_out, qk3_out, kq1_out, kq2_out, kq3_out, kk1_out, kk2_out, kk3_out, qq1_out, qq2_out, qq3_out = output_layers
            qks_emb = self.edge_bert_layer(qself_out,qk1_out,qk2_out,qk3_out)
            kqs_emb = self.edge_bert_layer(kself_out,kq1_out,kq2_out,kq3_out)
            qqs_emb = self.edge_bert_layer(qself_out,qq1_out,qq2_out,qq3_out)
            kks_emb = self.edge_bert_layer(kself_out,kk1_out,kk2_out,kk3_out)
            output_layer = torch.cat([output_layer_qk,qks_emb,kqs_emb,qqs_emb,kks_emb], dim=-1)
        elif not self.use_kkqq and self.use_qk:
            qself_out, kself_out, qk1_out, qk2_out, qk3_out, kq1_out, kq2_out, kq3_out = output_layers
            qks_emb = self.edge_bert_layer(qself_out, qk1_out, qk2_out, qk3_out)
            kqs_emb = self.edge_bert_layer(kself_out, kq1_out, kq2_out, kq3_out)
            output_layer = torch.cat([output_layer_qk, qks_emb, kqs_emb], dim=-1)
        elif self.use_kkqq and not self.use_qk:
            qself_out, kself_out, kk1_out, kk2_out, kk3_out, qq1_out, qq2_out, qq3_out = output_layers
            qqs_emb = self.edge_bert_layer(qself_out,qq1_out,qq2_out,qq3_out)
            kks_emb = self.edge_bert_layer(kself_out,kk1_out,kk2_out,kk3_out)
            output_layer = torch.cat([output_layer_qk, qqs_emb, kks_emb], dim=-1)
        else:
            raise ValueError
        # probilities setting
        _logits_ori = self.output_layer_ori(output_layer_qk)
        _logits_new = self.output_layer_edgebert(output_layer)

        if self.comb_loss:
            _logits = 0.2*_logits_ori + 0.8*_logits_new
        else:
            _logits = _logits_new

        # output part use transformers-style for comparison
        # original transformers use argmax for classification
        # for qk classification inference, see more in evaluate function
        # preds = F.softmax(_logits, dim=-1) # [bs, 2]
        # preds = preds[:, 1] # target score for roc_auc


        # update loss
        outputs = (_logits,)

        if label_id is not None:
            if self.num_labels == 1:
                # regression
                loss_fct = MSELoss()
                loss = loss_fct(_logits.view(-1), label_id.view(-1))
            else:
                # classification
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(_logits.view(-1, self.num_labels), label_id.view(-1))
            outputs = (loss,) + outputs
        return outputs

