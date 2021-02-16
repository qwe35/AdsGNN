# -*- coding: UTF-8 -*-

"""
Convert the QK data to token net

Include following parts
1. extract bert word embedding
    for the transformers bert embedding layer, it contains three part, and they are all nn.Embedding object
        - word_embeddings
        - position_embeddings
        - token_type_embeddings
    we only care the word_embeddings part
    Output:
        emebdding weight data (np.array)
        Loading example
            np.save('test3.npy', a)
            d = np.load('test3.npy')

2. generate the token net with networkx
    For <Q, K>, generate the token-net with networkx
    Original dataset
        schema :rid,label,query,doc,taskid
        for label: the mapping rule is [0，1，2，3] 0 is 0，else 1, only care non-0 data
    Input:
        graph_file_path
        model_dir_or_name : for tokenizer use
        graph_edge_mode: ["simple", "full"]
            - simple: do not differ the edge type
            - full: three edge types [1,2,3] for some HIN graph algos
        output_folder:
            save the nx object
3. merge different token-net from several sources
    generate offline sampling training dataset for token-net model

4. convert_networkx_json_format (optional)
    convert the networkx graph to json format for large graph engine
    Input
        G
        feature_data: from step 1
    Output
        data.json
        meta.json
"""

import torch
import os
import pickle
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
import copy
import random

def extract_bert_word_embedding(model_dir_or_name="bert-base-uncased", output_folder="../dummy_data/token_net/"):

    output_path = output_folder+"/"+"{}.token.feature.npy".format(model_dir_or_name.split("/")[-1])
    # load original
    bertmodel = BertModel.from_pretrained('bert-base-uncased')
    word_embedding_layer = bertmodel.embeddings.word_embeddings

    # save embedding
    saved_np_data = word_embedding_layer.weight.data.numpy()
    np.save(output_path, saved_np_data)
    print("save transformers {} bert word embedding \n output path is : {}".format(model_dir_or_name, output_path))


def add_networkx_edge_feature(G:nx.Graph, a:int, b:int, feature_name="weight", feature_value=1.0):
    if a not in G:
        G.add_node(a)
    if b not in G:
        G.add_node(b)
    if G.has_edge(a, b):
        if feature_name not in G[a][b]:
            G[a][b][feature_name] = feature_value
        else:
            G[a][b][feature_name] += feature_value
    else:
        G.add_edges_from([(a, b, {feature_name: feature_value})])


def generate_token_net_from_original_simple_edge_mode(
    graph_file_path,
    tokenizer
):
    """
    the simple mode to load original qk data
    use networkx Undirected Graph
    Only one type for edge
    """
    print("Vocab size is : {}".format(tokenizer.vocab_size))
    print("Load original qk dataset : {}".format(graph_file_path))
    G = nx.Graph()
    # init nodes
    G.add_nodes_from(list(range(tokenizer.vocab_size))) # then all nodes will in the graph

    bad_line_cnt = 0
    total_line_cnt = 0
    with open(graph_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            total_line_cnt += 1
            line_list = line.strip().split("\t")
            # original rid,label,query,doc,taskid
            if len(line_list) != 5:
                bad_line_cnt += 1
                continue
            label = int(line_list[1])
            query = line_list[2]
            keyword = line_list[3]
            if label > 0:
                q_list = tokenizer.encode(query, add_special_tokens=False)
                k_list = tokenizer.encode(keyword, add_special_tokens=False)
                for q in q_list:
                    for k in k_list:
                        add_networkx_edge_feature(G=G, a=q, b=k)
    print("Bad lines {} / {} ".format(bad_line_cnt, total_line_cnt))
    return G


def generate_token_net_from_original_full_edge_mode(
    graph_file_path,
    tokenizer
):
    """
    the full mode to load original qk data
    use networkx Undirected Graph
    Three types for edge
    """
    print("Vocab size is : {}".format(tokenizer.vocab_size))
    print("Load original qk dataset : {}".format(graph_file_path))
    G = nx.Graph()
    # init nodes
    G.add_nodes_from(list(range(tokenizer.vocab_size))) # then all nodes will in the graph

    bad_line_cnt = 0
    total_line_cnt = 0
    with open(graph_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            total_line_cnt += 1
            line_list = line.strip().split("\t")
            # original_dataset schema:  rid,label,query,doc,taskid
            if len(line_list) != 5:
                bad_line_cnt += 1
                continue
            label = int(line_list[1])
            query = line_list[2]
            keyword = line_list[3]
            if label > 0:
                q_list = tokenizer.encode(query, add_special_tokens=False)
                k_list = tokenizer.encode(keyword, add_special_tokens=False)
                for q in q_list:
                    for k in k_list:
                        add_networkx_edge_feature(G=G, a=q, b=k, feature_name="weight_{}".format(label))
    print("Bad lines {} / {} ".format(bad_line_cnt, total_line_cnt))
    return G

def generate_token_net_from_original_click_edge_mode(
    graph_file_path,
    tokenizer
):
    """
    the simple mode to load original qk data
    use networkx Undirected Graph
    Only one type for edge
    """
    print("Vocab size is : {}".format(tokenizer.vocab_size))
    print("Load original qk dataset : {}".format(graph_file_path))
    G = nx.Graph()
    # init nodes
    G.add_nodes_from(list(range(tokenizer.vocab_size))) # then all nodes will be in the graph

    bad_line_cnt = 0
    total_line_cnt = 0
    with open(graph_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            total_line_cnt += 1
            line_list = line.strip().split("\t")
            # query,doc,click
            if len(line_list) != 3:
                bad_line_cnt += 1
                continue
            query = line_list[0]
            keyword = line_list[1]
            click_num = int(line_list[2])
            q_list = tokenizer.encode(query, add_special_tokens=False)
            k_list = tokenizer.encode(keyword, add_special_tokens=False)
            for q in q_list:
                for k in k_list:
                    add_networkx_edge_feature(G=G, a=q, b=k, feature_value=click_num)
    print("Bad lines {} / {} ".format(bad_line_cnt, total_line_cnt))
    return G



def generate_token_net_from_original(
    graph_file_path,
    model_dir_or_name="bert-base-uncased",
    graph_edge_mode="simple",
    output_graph_path="../dummy_data/token_net/token_net.pkl"
):
    print("Load bert tokenizer : {}".format(model_dir_or_name.split("/")[-1]))
    tokenizer = BertTokenizer.from_pretrained(model_dir_or_name)


    print("Graph edge mode is {}".format(graph_edge_mode))
    if graph_edge_mode == "simple":
        G = generate_token_net_from_original_simple_edge_mode(graph_file_path=graph_file_path, tokenizer=tokenizer)
    elif graph_edge_mode == "full":
        G = generate_token_net_from_original_full_edge_mode(graph_file_path=graph_file_path, tokenizer=tokenizer)
    elif graph_edge_mode == "click":
        G = generate_token_net_from_original_click_edge_mode(graph_file_path=graph_file_path, tokenizer=tokenizer)
    else:
        raise NotImplementedError

    print("save the networkx in : {}".format(output_graph_path))
    with open(output_graph_path, 'wb') as f:
        pickle.dump(G, f)

    return G

def convert_networkx_graph_to_json_format(G: nx.Graph, feature_data, json_mode="simple", use_weight=False,output_folder="../dummy_data/token_net/json/"):
    """
    Output two data files
        - data.json
        - meta.json
    :param G:
    :param feature_data: extract_bert_word_embedding
    :param json_mode:
    :param output_folder:
    :return:
    """
    total_node_cnt = G.order()
    feature_dim = feature_data.shape[1]
    if total_node_cnt != feature_data.shape[0]:
        raise ValueError("graph and feature data not match")
    output_file_data_path = output_folder + "/" + "token_net_json_{}_mode_dim{}.json".format(json_mode,feature_dim)
    output_file_meta_path = output_folder + "/" + "meta.json".format(feature_dim)
    if os.path.exists(output_folder) is False:
        print("Make dir for : {}".format(output_folder))
        os.makedirs(output_folder)

    print("Process meta.json : {}".format(output_file_meta_path))

    if json_mode == "simple":
        meta = '{"node_float_feature_num": 1, \
                 "edge_binary_feature_num": 0, \
                 "edge_type_num": 1, \
                 "edge_float_feature_num": 0, \
                 "node_type_num": 1, \
                 "node_uint64_feature_num": 0, \
                 "node_binary_feature_num": 0, \
                 "edge_uint64_feature_num": 0}'
    elif json_mode == "full":
        """
        in full json_mode
        - there are three type of the edges for label[1,2,3]
        """
        meta = '{"node_float_feature_num": 1, \
                         "edge_binary_feature_num": 0, \
                         "edge_type_num": 3, \
                         "edge_float_feature_num": 0, \
                         "node_type_num": 1, \
                         "node_uint64_feature_num": 0, \
                         "node_binary_feature_num": 0, \
                         "edge_uint64_feature_num": 0}'
        #raise NotImplementedError
    else:
        raise NotImplementedError
    with open(output_file_meta_path, "w", encoding="utf=8") as f_out:
        f_out.write(meta)

    print("Process data.json : {}".format(output_file_data_path))
    with open(output_file_data_path, 'w', encoding='utf-8') as f_out:
        for node_idx in tqdm(range(total_node_cnt)):
            node_feature_list = feature_data[node_idx].tolist() # for the feature_data is np.array, which is not JSON serializable
            node_neighbor_list = list(G.neighbors(node_idx))


            if json_mode == "simple":

                # update node edge data info
                node_edge_info_list = []
                for neighbor_id in node_neighbor_list:
                    node_edge_info_list.append(
                        {
                            "src_id": node_idx,
                            "dst_id": neighbor_id,
                            "edge_type": 0,
                            "weight": G[node_idx][neighbor_id]["weight"],
                            "uint64_feature": {},
                            "float_feature": {},
                            "binary_feature": {}
                        }
                    )

                node = {
                    "node_weight": 1.0,
                    "node_id": node_idx,
                    "node_type": 0,
                    "uint64_feature": {},
                    "float_feature": {
                        "0": node_feature_list,
                    },
                    "binary_feature": {},
                    "edge": node_edge_info_list,
                    "neighbor": {
                        "0": dict([(str(neighbor_id), 1.0 if use_weight is False else G[node_idx][neighbor_id]["weight"]) for neighbor_id in node_neighbor_list]),
                    },
                }
            elif json_mode == "full":
                """
                there are three types weight_1, weight_2, weight_3
                """

                edge_type_0_neighbor_list = []
                edge_type_1_neighbor_list = []
                edge_type_2_neighbor_list = []
                for neighbor_id in node_neighbor_list:
                    if "weight_3" in G[node_idx][neighbor_id]:
                        edge_type_0_neighbor_list.append((str(neighbor_id), G[node_idx][neighbor_id]["weight_3"]))
                    if "weight_1" in G[node_idx][neighbor_id]:
                        edge_type_1_neighbor_list.append((str(neighbor_id), G[node_idx][neighbor_id]["weight_1"]))
                    if "weight_2" in G[node_idx][neighbor_id]:
                        edge_type_2_neighbor_list.append((str(neighbor_id), G[node_idx][neighbor_id]["weight_2"]))

                node = {
                    "node_weight": 1,
                    "node_id": node_idx,
                    "node_type": 0,
                    "uint64_feature": {},
                    "float_feature": {
                        "0": node_feature_list,
                    },
                    "binary_feature": {},
                    "edge": [],
                    "neighbor": {
                        "0": dict(edge_type_0_neighbor_list),
                        "1": dict(edge_type_1_neighbor_list),
                        "2": dict(edge_type_2_neighbor_list),
                    },
                }
            else:
                raise NotImplementedError("Only support simple json mode")

            node_line = json.dumps(node) + "\n"
            f_out.write(node_line)
    return output_file_meta_path, output_file_data_path

def filter_graph_by_weight_keep_top_K(OrigianlG:[nx.Graph, nx.DiGraph], weight_name="weight", topk=50, in_place=False):
    """
    sort neighbor by edge weight
        keep at most topk neighbors
    data
    :param G: nx.Graph
    :param weight_name: default is "weight"
    :param topk: 50
    :return:
    """
    if in_place:
        G=OrigianlG
    else:
        G = copy.deepcopy(OrigianlG)
    # change to Directed graph
    if type(G) is not nx.DiGraph:
        G = nx.DiGraph(G)
    node_id_list = list(G.nodes())
    node_id_list.sort()
    graph_density = nx.density(G)
    print("Graph info \n node_cnt : {} ".format(len(node_id_list)))
    print("Graph density : {}".format(graph_density))

    def return_edge_weight(node_id, tgt_id):
        return G[node_id][tgt_id][weight_name]

    for node_id in tqdm(node_id_list):
        node_neighbor_list = list(G.neighbors(node_id))
        # sort by weight
        node_neighbor_list.sort(key=lambda tgt_id: G[node_id][tgt_id][weight_name], reverse=True) # from big to small
        # remove edges
        if topk > len(node_neighbor_list):
            # just for safety check, actually unusefule
            continue
        for tgt_id in node_neighbor_list[topk:]:
            G.remove_edge(node_id, tgt_id)
    # return new Graph object
    # non-inline remove
    # return the Undirected graph
    G = G.to_undirected()
    return G


def generate_dummy_graph_data():
    """

    """
    import random
    output_file_path = "../dummy_data/dummy_edge_list.txt"
    node_list = list(range(300))
    edge_list_set = set()
    while len(edge_list_set) < 20000:
        tmp_left = random.choice(node_list)
        tmp_right = random.choice(node_list)
        if tmp_left == tmp_right:
            continue
        if (tmp_left, tmp_right) in edge_list_set:
            continue
        edge_list_set.add((tmp_left, tmp_right))
    edge_list = list(edge_list_set)
    # edge_list.sort(key=lambda edge: edge[0])
    with open(output_file_path, 'w', encoding="utf-8") as f:
        for edge in edge_list:
            f.write("{} {}\n".format(edge[0], edge[1]))


def sample_node_neighbor_networkx(G:nx.Graph, node_id:int, weight_name='weight', sample_cnt=10):
    if node_id not in G:
        return []
    neighbor_list = [x for x in G[node_id]]
    weight_list = [G[node_id][neigh_id][weight_name] for neigh_id in neighbor_list]
    sample_list = random.choices(
        population=neighbor_list,
        weights=weight_list,
        k=sample_cnt
    )
    return sample_list

class TokenNode:
    def __init__(self,
                 node_idx,
                 neigh_a_list,
                 neigh_b_list):
        self.node_idx=node_idx
        self.neigh_a_list=neigh_a_list
        self.neigh_b_list=neigh_b_list

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

def generate_token_node_from_networkx(G_a:nx.Graph, G_b:nx.Graph, node_idx:int, weight_name='weight', sample_cnt_a=10, sample_cnt_b=10):
    node_neigh_a_list = sample_node_neighbor_networkx(G_a, node_idx, weight_name=weight_name, sample_cnt=sample_cnt_a)
    node_neigh_b_list = sample_node_neighbor_networkx(G_b, node_idx, weight_name=weight_name, sample_cnt=sample_cnt_b)
    if len(node_neigh_a_list) == 0 or len(node_neigh_b_list) == 0:
        return None
    else:
        return TokenNode(
            node_idx=node_idx,
            neigh_a_list=node_neigh_a_list,
            neigh_b_list=node_neigh_b_list
        )

def merge_token_net_and_process_offline_sampling_dataset(
        token_net_a_path:str="../dummy_data/token_net/token_net.pkl",
        token_net_b_path:str="../dummy_data/token_net/token_net_click.pkl",
        output_folder:str="../dummy_data/token_net/offline_sampling/",
        neigh_cnt_a:int=10,
        neigh_cnt_b:int=10,
        negative_sample_n:int=3,
        weight_name='weight'
):
    # load networkx format graph
    with open(token_net_a_path, 'rb') as f:
        G_a = pickle.load(f)

    with open(token_net_b_path, 'rb') as f:
        G_b = pickle.load(f)

    node_list = sorted(list(G_a.nodes))
    token_node_dict = {}
    for idx in tqdm(node_list):
        token_node = generate_token_node_from_networkx(
            G_a=G_a,
            G_b=G_b,
            node_idx=idx,
            weight_name=weight_name,
            sample_cnt_a=neigh_cnt_a,
            sample_cnt_b=neigh_cnt_b
        )
        if token_node is not None:
            token_node_dict[idx] = token_node
    valid_node_idx_list = sorted(token_node_dict.keys())
    # generate unsupervised samples
    unsupervised_data_set = []
    for idx in tqdm(valid_node_idx_list):
        token_node_left = token_node_dict[idx]
        # positive sampling 
        token_node_right_idx = random.choice(token_node_left.neigh_a_list)
        token_node_right = token_node_dict[token_node_right_idx]
        unsupervised_data_set.append(
            UnsupervisedNode(
                node_left_idx=idx,
                node_right_idx=token_node_right_idx,
                node_left_neigh_a_list=token_node_left.neigh_a_list,
                node_left_neigh_b_list=token_node_left.neigh_b_list,
                node_right_neigh_a_list=token_node_right.neigh_a_list,
                node_right_neigh_b_list=token_node_right.neigh_b_list,
                label=1
            ).__dict__
        )
        # negative sampling
        for negative_idx in range(negative_sample_n):
            token_node_right_idx = random.choice(valid_node_idx_list)
            token_node_right = token_node_dict[token_node_right_idx]
            unsupervised_data_set.append(
                UnsupervisedNode(
                    node_left_idx=idx,
                    node_right_idx=token_node_right_idx,
                    node_left_neigh_a_list=token_node_left.neigh_a_list,
                    node_left_neigh_b_list=token_node_left.neigh_b_list,
                    node_right_neigh_a_list=token_node_right.neigh_a_list,
                    node_right_neigh_b_list=token_node_right.neigh_b_list,
                    label=0
                ).__dict__
            )
    # output dataset
    print("ouput dataset : {}".format(output_folder))
    if os.path.exists(output_folder) is None:
        os.makedirs(output_folder)
    output_path_valid = output_folder + '/' + 'valid_id.pkl'
    with open(output_path_valid, 'wb') as f:
        pickle.dump(valid_node_idx_list, f)
    output_path_data = output_folder + '/' + 'unsupervised_dataset.pkl'
    with open(output_path_data, 'wb') as f:
        pickle.dump(unsupervised_data_set, f)


def test_built_graph_demo_relevance():
    graph_file_path = "../dummy_data/QKTrain.tsv"
    G = generate_token_net_from_original(
        graph_file_path,
        output_graph_path="../dummy_data/token_net/token_net.pkl"
    )
def test_built_graph_demo_click():
    graph_file_path = "../dummy_data/click_data/click_data_log.tsv"
    G = generate_token_net_from_original(
        graph_file_path,
        output_graph_path="../dummy_data/token_net/token_net_click.pkl",
        graph_edge_mode="click"
    )

def test_filter_graph():
    topk = 30
    graph_path = "../dummy_data/token_net/token_net_click.pkl"
    output_graph_path = "../dummy_data/token_net/token_net_click_with_click_weight_top30.pkl"

    print(f'original graph path : {graph_path} \n'\
          f'output graph path : {output_graph_path} \n'\
          f'top {topk} weight neighbors'
          )
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    newG = filter_graph_by_weight_keep_top_K(OrigianlG=G, topk=topk)
    with open(output_graph_path, 'wb') as f:
        pickle.dump(newG, f)

def test_convert_json():
    # load graph and feature data
    graph_path = "../dummy_data/token_net/token_net_click_with_click_weight_top30.pkl"
    feature_data = np.load("../dummy_data/token_net/bert-base-uncased.token.feature.npy")
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    json_output_folder = "../dummy_data/token_net/token_net_click/"
    output_file_meta_path, output_file_data_path = convert_networkx_graph_to_json_format(G, feature_data, json_mode="simple", use_weight=False, output_folder=json_output_folder)
    # then test the processed json files
    with open(output_file_data_path, 'r', encoding="utf-8") as f:
        cnt = 0
        for line in tqdm(f):
            cnt += 1
            if cnt > 3200:
                json_graph_data = line.strip()
                break
    node = json.loads(json_graph_data)
    print(node)

if __name__ == '__main__':
    # extract_bert_word_embedding()
    # test_built_graph_demo_relevance()
    # test_built_graph_demo_click()
    # test_filter_graph()
    test_convert_json()
    # test_json_data_file()
