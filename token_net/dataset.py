# -*- coding: UTF-8 -*-

"""
for dataset generation, see convert_to_token_net.py
"""
import pickle
import torch
from torch.utils.data import TensorDataset

class UnsupervisedNode:
    def __init__(self,
                 node_left_idx,
                 node_right_idx,
                 node_left_neigh_a_list,
                 node_left_neigh_b_list,
                 node_right_neigh_a_list,
                 node_right_neigh_b_list,
                 label
                 ):
        self.node_left_idx = node_left_idx
        self.node_right_idx = node_right_idx
        self.node_left_neigh_a_list = node_left_neigh_a_list
        self.node_left_neigh_b_list = node_left_neigh_b_list
        self.node_right_neigh_a_list = node_right_neigh_a_list
        self.node_right_neigh_b_list = node_right_neigh_b_list
        self.label = label


def convert_examples_to_features(data_dir):
    """
    output_path_data = output_folder + '/' + 'unsupervised_dataset.pkl'
    """
    dataset_path = data_dir + '/' + 'unsupervised_dataset.pkl'
    with open(dataset_path, 'rb') as f:
        unsupervised_data_set = pickle.load(f)
    feature_list = []
    for unsupervised_data_node in unsupervised_data_set:
        feature_list.append(UnsupervisedNode(**unsupervised_data_node))
    return feature_list


def convert_to_Tensors(features):
    node_left_idx = torch.tensor([f.node_left_idx for f in features], dtype=torch.long)
    node_right_idx = torch.tensor([f.node_right_idx for f in features], dtype=torch.long)
    node_left_neigh_a_list = torch.tensor([f.node_left_neigh_a_list for f in features], dtype=torch.long)
    node_left_neigh_b_list = torch.tensor([f.node_left_neigh_b_list for f in features], dtype=torch.long)
    node_right_neigh_a_list = torch.tensor([f.node_right_neigh_a_list for f in features], dtype=torch.long)
    node_right_neigh_b_list = torch.tensor([f.node_right_neigh_b_list for f in features], dtype=torch.long)
    label = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        node_left_idx,
        node_right_idx,
        node_left_neigh_a_list,
        node_left_neigh_b_list,
        node_right_neigh_a_list,
        node_right_neigh_b_list,
        label
    )
    return dataset

def generate_model_inputs(batch):
    # feature format
    # features = {
    #     'node_left_idx':batch[0],
    #     'node_right_idx':batch[1],
    #     'node_left_neigh_a_list':batch[2],
    #     'node_left_neigh_b_list':batch[3],
    #     'node_right_neigh_a_list':batch[4],
    #     'node_right_neigh_b_list':batch[5],
    #     'label':batch[-1]
    # }
    node_left = {
        'node_feat': batch[0],
        'neigh_a_feat_list': batch[2],
        'neigh_b_feat_list': batch[3]
    }
    node_right = {
        'node_feat': batch[1],
        'neigh_a_feat_list': batch[4],
        'neigh_b_feat_list': batch[5]
    }

    output_ret = {
        'node_left':node_left,
        'node_right':node_right,
        'label':batch[-1]
    }
    return output_ret