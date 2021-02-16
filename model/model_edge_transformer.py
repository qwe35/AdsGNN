# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers.modeling_bert import BertEncoder, BertPooler
BertLayerNorm = torch.nn.LayerNorm

"""
similar to model edge_conv
use transformer-based gnn model
use https://github.com/huggingface/transformers as the Bert encoder
"""

class ModelQKEdgeBERTConfig(object):
    """
    config file for ModelQKEdgeBERTConfig
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
                 num_hidden_layers=2,
                 max_inti_pos_index=16,
                 graph_bert_hidden_size=768,
                 num_attention_heads=8,
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

        # parameter for the transformer layer setting
        self.graph_bert_config=BertConfig.from_pretrained('bert-base-uncased')
        # update the bert_config setting
        self.graph_bert_config.hidden_size=graph_bert_hidden_size
        self.graph_bert_config.num_hidden_layers=num_hidden_layers
        self.graph_bert_config.max_inti_pos_index=max_inti_pos_index
        self.graph_bert_config.num_attention_heads=num_attention_heads

class GraphBertEmbeddings(nn.Module):
    """Construct the embeddings from features, wl, position and hop vectors.
    """

    def __init__(self, config):
        super(GraphBertEmbeddings, self).__init__()
        self.inti_pos_embeddings = nn.Embedding(config.max_inti_pos_index, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, raw_feature_embeds=None, init_pos_ids=None):
        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        #---- here, we use summation ----
        embeddings = raw_feature_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GraphBertBlock(BertPreTrainedModel):
    def __init__(self, config):
        super(GraphBertBlock, self).__init__(config)
        self.config = config
        self.embeddings = GraphBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, raw_feature_embeds, init_pos_ids,
                output_attentions=None, output_hidden_states=None, head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                attention_mask=None,
                token_type_ids=None,
                ):
        """
        Input
            raw_feature_embeds : [torch.FloatTensor], [bs, seq_len, hidden_size]
            init_pos_ids: [torch.LongTensor], [bs, seq_len]
        Ouptut
            Return the tuple instead of simply tensor
            pooled sequence output [torch.FloatTensor]
            same as transformers.BertModel

            sequence_output, pooled_output, (hidden_states)
        """
        device = init_pos_ids.device if init_pos_ids is not None else raw_feature_embeds.device

        # embedding part
        embedding_output = self.embeddings(
            raw_feature_embeds=raw_feature_embeds, init_pos_ids=init_pos_ids
        )
        # encoder part
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )


        # Preprocess for Encoder Part
        # extended_attention_mask and head mask
        input_shape = init_pos_ids.size()
        # extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        # head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # update extended_attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Encoder part
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_extended_attention_mask,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states)

class ModelQKEdgeBERT(nn.Module):
    def __init__(self, args=None):
        """
        Contain following contents:
        - bert model (need to freeze parameter)
            - bert_model_path_or_name
        - edgebert model

        same as the EdgeBert model
            - qk bert to generate the text embedding feature
            - transformer to generate the crossing feature
        """
        super(ModelQKEdgeBERT, self).__init__()
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
        self.output_layer_qk_dropout = nn.Dropout(p=0.1) # for torch Dropout p – probability of an element to be zeroed. Default: 0.5
        # graph part
        # self.edgebert_hidden_size = args.edgebert_hidden_size
        self.use_kkqq = args.use_kkqq
        self.use_qk = args.use_qk
        # output layer
        self.output_layer_ori = nn.Linear(in_features=self.bert_hidden_size, out_features=args.num_labels, bias=True)
        # output for GraphBert
        self.graph_bert_hidden_size = args.graph_bert_config.hidden_size
        self.q_graph_bert = GraphBertBlock(args.graph_bert_config)
        self.k_graph_bert = GraphBertBlock(args.graph_bert_config)
        # transform layer from Bert
        self.graph_bert_transform_layer = nn.Linear(in_features=self.bert_hidden_size, out_features=self.graph_bert_hidden_size)
        if self.use_kkqq and self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.graph_bert_hidden_size*2, out_features=args.num_labels, bias=False)
        elif not self.use_kkqq and self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.graph_bert_hidden_size*2, out_features=args.num_labels, bias=False)
        elif self.use_kkqq and not self.use_qk:
            self.output_layer_edgebert = nn.Linear(in_features=self.bert_hidden_size+self.graph_bert_hidden_size*2, out_features=args.num_labels, bias=False)
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
        # use transformer layer to generate features
        batch_size = input_ids.shape[0]
        device = input_ids.device
        if self.use_kkqq and self.use_qk:
            qself_out, kself_out, qk1_out, qk2_out, qk3_out, kq1_out, kq2_out, kq3_out, kk1_out, kk2_out, kk3_out, qq1_out, qq2_out, qq3_out = output_layers
            seqeuece_cat_q = torch.stack([output_layer_qk,qk1_out,qk2_out,qk3_out], dim=1)
            seqeuece_cat_k = torch.stack([output_layer_qk,kq1_out,kq2_out,kq3_out], dim=1)
            # tranform for graph bert layer
            seqeuece_cat_q = self.graph_bert_transform_layer(seqeuece_cat_q)
            seqeuece_cat_k = self.graph_bert_transform_layer(seqeuece_cat_k)
            # attention mask and pos_index
            pos_index_q = torch.LongTensor([0,1,1,1]).repeat([batch_size,1]).to(device)
            pos_index_k = torch.LongTensor([0,2,2,2]).repeat([batch_size,1]).to(device)
            graph_bert_attention_mask = torch.LongTensor([1,1,1,1]).repeat([batch_size,1]).to(device)
            graph_emb_q = self.q_graph_bert(raw_feature_embeds=seqeuece_cat_q, init_pos_ids=pos_index_q, attention_mask=graph_bert_attention_mask)
            graph_emb_k = self.q_graph_bert(raw_feature_embeds=seqeuece_cat_k, init_pos_ids=pos_index_k, attention_mask=graph_bert_attention_mask)
            pooled_graph_emb_q = graph_emb_q[1]
            pooled_graph_emb_k = graph_emb_k[1]
            output_layer = torch.cat([output_layer_qk, pooled_graph_emb_q, pooled_graph_emb_k], dim=-1)
        else:
            raise NotImplementedError

        _logits_ori = self.output_layer_ori(output_layer_qk)
        _logits_new = self.output_layer_edgebert(output_layer)

        if self.comb_loss:
            _logits = 0.2 * _logits_ori + 0.8 * _logits_new
        else:
            _logits = _logits_new

        # output part is transformers-style for comparison
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