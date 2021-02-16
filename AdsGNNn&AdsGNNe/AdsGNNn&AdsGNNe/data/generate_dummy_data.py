# -*- coding: UTF-8 -*-

import random
import os
import shutil
from tqdm import tqdm

"""
generate dummy dataset for this repo 
"""

def generate_dummy_dataset_qk33gnn(output_folder, train_cnt=256, test_cnt=32):
    """
    schema: [rid,label,query,doc,k1,k2,k3,q1,q2,q3,task_id]
    tsv file
    task_id : for future multi-task learning use
    for qk33gnn processor
    :return:
    """
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)


    tmp_txt = "start your engine"

    tmp_line = '\t'.join([tmp_txt]*8)
    tmp_line = tmp_line + '\t' + '0'
    # query,doc,k1,k2,k3,q1,q2,q3,task_id

    train_file_path = output_folder + '/' + 'QKTrain.tsv'
    val_file_path = output_folder + '/' + 'QKVal.tsv'
    test_file_path = output_folder + '/' + 'QKTest.tsv'
    if os.path.exists(train_file_path):
        raise FileExistsError('file exists {}'.format(train_file_path))

    with open(train_file_path, 'w', encoding='utf-8') as f:
        for rid in range(train_cnt):
            cur_label = random.randint(0, 1)
            cur_line = '{}\t{}\t{}'.format(rid, cur_label, tmp_line)
            f.write(cur_line + '\n')

    with open(val_file_path, 'w', encoding='utf-8') as f:
        for rid in range(test_cnt):
            cur_label = random.randint(0, 1)
            cur_line = '{}\t{}\t{}'.format(rid, cur_label, tmp_line)
            f.write(cur_line + '\n')

    with open(test_file_path, 'w', encoding='utf-8') as f:
        for rid in range(test_cnt):
            cur_label = random.randint(0, 1)
            cur_line = '{}\t{}\t{}'.format(rid, cur_label, tmp_line)
            f.write(cur_line + '\n')


def generate_dummy_dataset_qk(output_folder, train_cnt=256, test_cnt=32):
    """
    schema: [rid,label,query,doc,taskid]
    tsv file
    task_id : for future multi-task learning use
    for qk processor
    :return:
    """
    if os.path.exists(output_folder) is False:
        os.makedirs(output_folder)

    tmp_txt = "start your engine"

    tmp_line = '\t'.join([tmp_txt] * 2)
    tmp_line = tmp_line + '\t' + '0'  # query,doc,k1,k2,k3,q1,q2,q3,task_id

    train_file_path = output_folder + '/' + 'QKTrain.tsv'
    val_file_path = output_folder + '/' + 'QKVal.tsv'
    test_file_path = output_folder + '/' + 'QKTest.tsv'
    if os.path.exists(train_file_path):
        raise FileExistsError('file exists {}'.format(train_file_path))

    with open(train_file_path, 'w', encoding='utf-8') as f:
        for rid in range(train_cnt):
            cur_label = random.randint(0, 1)
            cur_line = '{}\t{}\t{}'.format(rid, cur_label, tmp_line)
            f.write(cur_line + '\n')

    with open(val_file_path, 'w', encoding='utf-8') as f:
        for rid in range(test_cnt):
            cur_label = random.randint(0, 1)
            cur_line = '{}\t{}\t{}'.format(rid, cur_label, tmp_line)
            f.write(cur_line + '\n')

    with open(test_file_path, 'w', encoding='utf-8') as f:
        for rid in range(test_cnt):
            cur_label = random.randint(0, 1)
            cur_line = '{}\t{}\t{}'.format(rid, cur_label, tmp_line)
            f.write(cur_line + '\n')


def qk33gnn_demo():
    output_folder = "../dummy_data/qk33gnn320/"
    generate_dummy_dataset_qk33gnn(output_folder)

def qk_demo():
    output_folder = "../dummy_data/qk320/"
    generate_dummy_dataset_qk(output_folder)


def split_dataset_file(data_folder:str,
                       split_ratio:float,
                       rand=random.Random(),
                       data_FN='QKTrain.tsv',
                       val_FN='QKVal.tsv',
                       test_FN='QKTest.tsv'
                       ):
    """
    create dataset for training ratio experiments
        10%, 30%, 50%, 70%, 100%
    for semi-supervised machine learning
        test model on smaller training dataset
    use random
    :param data_folder:
    :param split_ratio: like 0.1
    """
    assert split_ratio <= 1
    assert split_ratio > 0
    input_data_path = data_folder + '/' + data_FN
    output_folder = data_folder + '/train_split_{}/'.format(split_ratio)
    if os.path.exists(output_folder):
        raise FileExistsError("{} exists".format(output_folder))
    else:
        os.makedirs(output_folder)
    output_data_path = output_folder + '/' + data_FN

    print("*" * 30)
    print("load dataset : {}".format(input_data_path))
    print("split ratio is {}".format(split_ratio))
    print("output path  : {}".format(output_data_path))
    print("*" * 30)

    output_file = open(output_data_path, 'w', encoding='utf-8')

    with open(input_data_path, 'r', encoding='utf-8') as input_file:
        for line in tqdm(input_file):
            # generate the rand
            line_rand = rand.random()
            if line_rand < split_ratio:
                new_line = line.strip() + '\n'
                output_file.write(new_line)

    output_file.close()

    val_data_path = data_folder + '/' + val_FN
    if os.path.isfile(val_data_path):
        shutil.copy(val_data_path, output_folder)

    test_data_path = data_folder + '/' + test_FN
    if os.path.isfile(test_data_path):
        shutil.copy(test_data_path, output_folder)



def split_dataset_demo():

    data_folder_list = [
        "../dummy_data/qk320/",
        "../dummy_data/qk33gnn320/",
    ]
    split_ratio_list = [
        0.1, 0.3, 0.5, 0.7
    ]

    for data_folder in data_folder_list:
        for split_ratio in split_ratio_list:
            split_dataset_file(data_folder,split_ratio)


def split_dataset_with_guideline(
        data_folder:str,
        guide_folder:str,
        split_ratio:float,
        data_FN='QKTrain.tsv',
        val_FN='QKVal.tsv',
        test_FN='QKTest.tsv'):
    """
    create dataset for training ratio experiments
        10%, 30%, 50%, 70%, 100%
    for semi-supervised machine learning
        test model on less training dataset
    instead of random split use guidline file
        with guid of data
    """

    assert split_ratio <= 1
    assert split_ratio > 0
    input_data_path = data_folder + '/' + data_FN
    output_folder = data_folder + '/train_split_{}/'.format(split_ratio)
    if os.path.exists(output_folder):
        raise FileExistsError("{} exists".format(output_folder))
    else:
        os.makedirs(output_folder)
    output_data_path = output_folder + '/' + data_FN

    guide_data_path = guide_folder+'/train_split_{}/'.format(split_ratio) + '/' + data_FN
    if os.path.exists(guide_data_path) is False:
        raise FileNotFoundError("{} not found".format(guide_data_path))

    print("*" * 30)
    print("load dataset : {}".format(input_data_path))
    print("split ratio is {}".format(split_ratio))
    print("output path  : {}".format(output_data_path))
    print("guid data path: {}".format(guide_data_path))
    print("*" * 30)

    # start to load guide guid set
    select_guid_set = set()
    with open(guide_data_path, 'r', encoding='utf-8') as guide_file:
        for line in guide_file:
            line_list = line.strip().split('\t')
            line_guid = int(line_list[0])
            select_guid_set.add(line_guid)

    # start to split file
    output_file = open(output_data_path, 'w', encoding='utf-8')

    with open(input_data_path, 'r', encoding='utf-8') as input_file:
        for line in tqdm(input_file):
            # generate the rand
            line_list = line.strip().split('\t')
            line_guid = int(line_list[0])
            if line_guid in select_guid_set:
                new_line = line.strip() + '\n'
                output_file.write(new_line)

    output_file.close()

    val_data_path = data_folder + '/' + val_FN
    if os.path.isfile(val_data_path):
        shutil.copy(val_data_path, output_folder)

    test_data_path = data_folder + '/' + test_FN
    if os.path.isfile(test_data_path):
        shutil.copy(test_data_path, output_folder)

def split_dataset_with_guideline_demo():
    data_folder = "../dummy_data/qk320/"
    guide_folder = "../dummy_data/qk33gnn320/"
    split_ratio_list = [
        0.1, 0.3, 0.5, 0.7
    ]

    for split_ratio in split_ratio_list:
        split_dataset_with_guideline(
            data_folder=data_folder,
            guide_folder=guide_folder,
            split_ratio=split_ratio
        )

if __name__ == '__main__':
    qk33gnn_demo()
    qk_demo()
    split_dataset_demo()
    #split_dataset_with_guideline_demo()
    pass