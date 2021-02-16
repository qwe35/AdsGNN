# -*- coding: UTF-8 -*-
"""
model for token net model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class AttnHead(nn.Module):
    """
    self attention head for gnn model
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
        out = sum(hidden) / self.head_nums
        return out # same shape as seq

class AttentionLayerConfig:
    def __init__(self,
                 num_attention_heads=4,
                 hidden_size=1024,
                 attention_probs_dropout_prob=0.1,
                 ):
        self.num_attention_heads=num_attention_heads
        self.hidden_size=hidden_size
        self.attention_probs_dropout_prob=attention_probs_dropout_prob

class AttentionLayer(nn.Module):
    def __init__(self, config):
        super(AttentionLayer, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, node_feats, neigh_feats):
        mixed_query_layer = self.query(node_feats)
        mixed_key_layer = self.key(neigh_feats)
        mixed_value_layer = self.value(neigh_feats)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Donot apply the attention mask for this model
        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # [bs, head_num, query_num, value_dim]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer # [bs, query_num, value_dim]

class TokenNetConfig:
    def __init__(self,
                 feature_dim:int=768,
                 head_nums:int=12,
                 use_self_attn=False,
                 attn_use_residual=True,
                 attn_layer_num=4,
                 attn_layer_drop_out_rate=0.2,
                 attn_layer_head_num=4,
                 feature_data_path=None
                 ):
        self.feature_dim=feature_dim
        self.head_nums=head_nums
        self.use_self_attn=use_self_attn
        self.attn_act=F.relu
        self.attn_use_residual=attn_use_residual
        self.attn_layer_num=attn_layer_num
        self.attn_layer_drop_out_rate=attn_layer_drop_out_rate
        self.attn_layer_head_num=attn_layer_head_num

        self.feature_data_path=feature_data_path

class TokenNet(nn.Module):
    def __init__(self,
                 config:TokenNetConfig):
        super(TokenNet, self).__init__()
        self.use_self_attn = config.use_self_attn
        if config.use_self_attn:
            self.multiattn_a = MultiAttnHead(head_nums=config.head_nums, input_size=config.feature_dim,
                                           out_size=config.feature_dim,
                                           act_fn=config.attn_act, residual=config.attn_use_residual)
            self.multiattn_b = MultiAttnHead(head_nums=config.head_nums, input_size=config.feature_dim,
                                             out_size=config.feature_dim,
                                             act_fn=config.attn_act, residual=config.attn_use_residual)

        # local attention
        self.attn_layer_a = AttentionLayer(AttentionLayerConfig(
            num_attention_heads=config.attn_layer_head_num,
            hidden_size=config.feature_dim,
            attention_probs_dropout_prob=config.attn_layer_drop_out_rate
        ))
        self.attn_layer_b = AttentionLayer(AttentionLayerConfig(
            num_attention_heads=config.attn_layer_head_num,
            hidden_size=config.feature_dim,
            attention_probs_dropout_prob=config.attn_layer_drop_out_rate
        ))

        # global attention
        self.attn_layer_global = AttentionLayer(AttentionLayerConfig(
            num_attention_heads=config.attn_layer_head_num,
            hidden_size=config.feature_dim,
            attention_probs_dropout_prob=config.attn_layer_drop_out_rate
        ))

    def forward(self, node_feat, neigh_a_feat_list, neigh_b_feat_list):
        """
        Input
            node_feat : torch.tensor float, [bs, feat_dim]
            neigh_a_feat_list: list of torch.tensor, from relevance graph, e.g., list of (bs, feat_dim)
            neigh_b_feat_list: list of torch.tensor, from relevance graph, e.g., list of (bs, feat_dim)
        """
        assert node_feat.dim() == 2
        assert neigh_a_feat_list[0].dim() == 2

        node_feat = node_feat.unsqueeze(dim=1) # [bs, 1, feat_dim]
        neigh_a_feat_list = [feat.unsqueeze(dim=1) for feat in neigh_a_feat_list]
        neigh_b_feat_list = [feat.unsqueeze(dim=1) for feat in neigh_b_feat_list]

        neigh_a_feat = torch.cat(neigh_a_feat_list, dim=1) # [bs, neigh_a_cnt, feat_dim]
        neigh_b_feat = torch.cat(neigh_b_feat_list, dim=1) # [bs, neigh_a_cnt, feat_dim]

        # self-attn
        if self.use_self_attn:
            neigh_a_feat = self.multiattn_a(torch.cat([neigh_a_feat, node_feat], dim=1))
            neigh_b_feat = self.multiattn_b(torch.cat([neigh_b_feat, node_feat], dim=1))

        # local attention
        node_neigh_a_feat = self.attn_layer_a(node_feat, neigh_a_feat) # [bs, 1, feat_dim]
        node_neigh_b_feat = self.attn_layer_b(node_feat, neigh_b_feat) # [bs, 1, feat_dim]

        # global attention
        node_neigh_global_feat = torch.cat([node_neigh_a_feat, node_feat, node_neigh_b_feat], dim=1)
        out_feat = self.attn_layer_global(node_feat, node_neigh_global_feat)
        out_feat = out_feat.squeeze()
        return out_feat # [bs, feat_dim]


class TokenNetEncoder(nn.Module):
    def __init__(self, config:TokenNetConfig):
        super(TokenNetEncoder, self).__init__()
        if config.feature_data_path is None:
            self.feat_embedding_layer = nn.Embedding(1024, 768)
        else:
            feature_data = np.load(config.feature_data_path)
            assert feature_data.ndim == 2
            node_cnt, feature_dim = feature_data.shape
            assert feature_dim == config.feature_dim
            self.feat_embedding_layer = nn.Embedding(node_cnt, feature_dim)
            self.feat_embedding_layer.weight == nn.Parameter(feature_data)
            self.feat_embedding_layer.weight.requires_grad = False

        self.token_net = TokenNet(config)

    def forward(self, node_feat, neigh_a_feat_list, neigh_b_feat_list):
        node_feat = self.feat_embedding_layer(node_feat)
        neigh_a_feat_list = [self.feat_embedding_layer(neigh_node) for neigh_node in neigh_a_feat_list]
        neigh_b_feat_list = [self.feat_embedding_layer(neigh_node) for neigh_node in neigh_b_feat_list]

        out = self.token_net(
            node_feat, neigh_a_feat_list, neigh_b_feat_list
        )
        return out

class UnsupervisedTokenNet(nn.Module):
    """
    UnsupervisedModel for TokenNet Training
    """
    def __init__(self,
                 config:TokenNetConfig
                 ):
        super(UnsupervisedTokenNet, self).__init__()
        self.encoder = TokenNetEncoder(config)
        self.clf = nn.Linear(in_features=config.feature_dim*2, out_features=2)

    def forward(self, node_left:dict, node_right:dict, label:torch.tensor=None):
        node_left_emb = self.encoder(
            node_feat=node_left['node_feat'],
            neigh_a_feat_list=node_left['neigh_a_feat_list'],
            neigh_b_feat_list=node_left['neigh_b_feat_list']
        )

        node_right_emb = self.encoder(
            node_feat=node_right['node_feat'],
            neigh_a_feat_list=node_right['neigh_a_feat_list'],
            neigh_b_feat_list=node_right['neigh_b_feat_list']
        )

        node_pair_emb = torch.cat([node_left_emb, node_right_emb], dim=1)
        logits = self.clf(node_pair_emb)

        output = (logits,)

        if label:
            loss = torch.nn.functional.cross_entropy(logits.view(-1), label.view(-1))
            output = (loss,) + output

        return output

def test_attn_layer_demo():
    node_feats = torch.ones([32, 2, 1024])

    neigh_feats = torch.ones([32, 8, 1024])

    attn_config = AttentionLayerConfig(num_attention_heads=16)
    attn_model = AttentionLayer(attn_config)

    context_layer = attn_model(node_feats, neigh_feats)
    print(context_layer.shape)


if __name__ == '__main__':
    feat_list = [torch.ones([32,1024]) for _ in range(8)]
    feat_list = [f.unsqueeze(dim=1) for f in feat_list]
    feat = torch.cat(feat_list, dim=1)
    print(feat.shape)


    node_idx = torch.ones([32]).long()
    node_a_list = [torch.ones([32]).long()]*8
    node_b_list = [torch.ones([32]).long()]*8

    node_dict = {
        'node_feat': node_idx,
        'neigh_a_feat_list':node_a_list,
        'neigh_b_feat_list':node_b_list
    }

    config = TokenNetConfig()
    model = UnsupervisedTokenNet(config)
    out = model(node_dict, node_dict, label=None)