# -*- coding: UTF-8 -*-
"""
data processor for edge conv model
"""
import copy
import json
import csv
import os
import logging
from tqdm import tqdm
from .file_utils import is_tf_available
if is_tf_available():
    import tensorflow as tf


logger = logging.getLogger(__name__)


class InputExample_GNN(object):
    """
    A single training/test example for ads GNN dataset

    Args:
        guid: Unique id for the example.
        text_a: string. query for Ads GNN
        text_b: string. keyword for Ads GNN
        k1: string
        k2: string
        k3: string
        q1: string
        q2: string
        q3: string
        label: string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
        task: for further usage
    """
    def __init__(self, guid, text_a, text_b=None,k1=None,k2=None, k3=None,q1=None,q2=None,q3=None,label=None, task=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.q1 = q1
        self.q2 = q2
        self.q3 = q3

        self.label = label
        self.task = task

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures_GNN(object):
    """
    Input Feature class for baseline3
    A single set of features of data.
    Processed by tokenizer
        contains input_ids, attention_mask, token_type_ids
        - q
        - k
        - qk
        - qk1
        - qk2
        - qk3
        - kq1
        - kq2
        - kq3
        - qq1
        - qq2
        - qq3
        - kk1
        - kk2
        - kk3
    Other data:
        - row id: unique is for the feature
        - label_id: label(int or float)
        - task_id: for further usage
    """
    def __init__(self,
               input_ids_q,input_mask_q,segment_ids_q,
               input_ids_k,input_mask_k,segment_ids_k,
               input_ids_qk,input_mask_qk,segment_ids_qk,
               input_ids_qk1,input_mask_qk1,segment_ids_qk1,
               input_ids_qk2,input_mask_qk2,segment_ids_qk2,
               input_ids_qk3,input_mask_qk3,segment_ids_qk3,
               input_ids_kq1,input_mask_kq1,segment_ids_kq1,
               input_ids_kq2,input_mask_kq2,segment_ids_kq2,
               input_ids_kq3,input_mask_kq3,segment_ids_kq3,
               input_ids_qq1,input_mask_qq1,segment_ids_qq1,
               input_ids_qq2,input_mask_qq2,segment_ids_qq2,
               input_ids_qq3,input_mask_qq3,segment_ids_qq3,
               input_ids_kk1,input_mask_kk1,segment_ids_kk1,
               input_ids_kk2,input_mask_kk2,segment_ids_kk2,
               input_ids_kk3,input_mask_kk3,segment_ids_kk3,
               row_id,label_id,task_id,
               is_real_example=True):
        # frocessed for the edge sequence ids
        self.input_ids_q = input_ids_q
        self.input_mask_q = input_mask_q
        self.segment_ids_q = segment_ids_q

        self.input_ids_k = input_ids_k
        self.input_mask_k = input_mask_k
        self.segment_ids_k = segment_ids_k

        self.input_ids_qk = input_ids_qk
        self.input_mask_qk = input_mask_qk
        self.segment_ids_qk = segment_ids_qk

        self.input_ids_qk1 = input_ids_qk1
        self.input_mask_qk1 = input_mask_qk1
        self.segment_ids_qk1 = segment_ids_qk1

        self.input_ids_qk2 = input_ids_qk2
        self.input_mask_qk2 = input_mask_qk2
        self.segment_ids_qk2 = segment_ids_qk2

        self.input_ids_qk3 = input_ids_qk3
        self.input_mask_qk3 = input_mask_qk3
        self.segment_ids_qk3 = segment_ids_qk3

        self.input_ids_kq1 = input_ids_kq1
        self.input_mask_kq1 = input_mask_kq1
        self.segment_ids_kq1 = segment_ids_kq1

        self.input_ids_kq2 = input_ids_kq2
        self.input_mask_kq2 = input_mask_kq2
        self.segment_ids_kq2 = segment_ids_kq2

        self.input_ids_kq3 = input_ids_kq3
        self.input_mask_kq3 = input_mask_kq3
        self.segment_ids_kq3 = segment_ids_kq3

        self.input_ids_qq1 = input_ids_qq1
        self.input_mask_qq1 = input_mask_qq1
        self.segment_ids_qq1 = segment_ids_qq1

        self.input_ids_qq2 = input_ids_qq2
        self.input_mask_qq2 = input_mask_qq2
        self.segment_ids_qq2 = segment_ids_qq2

        self.input_ids_qq3 = input_ids_qq3
        self.input_mask_qq3 = input_mask_qq3
        self.segment_ids_qq3 = segment_ids_qq3

        self.input_ids_kk1 = input_ids_kk1
        self.input_mask_kk1 = input_mask_kk1
        self.segment_ids_kk1 = segment_ids_kk1

        self.input_ids_kk2 = input_ids_kk2
        self.input_mask_kk2 = input_mask_kk2
        self.segment_ids_kk2 = segment_ids_kk2

        self.input_ids_kk3 = input_ids_kk3
        self.input_mask_kk3 = input_mask_kk3
        self.segment_ids_kk3 = segment_ids_kk3

        self.row_id = row_id
        self.label_id = label_id
        self.is_real_example = is_real_example
        self.task_id = task_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures_GNN_Token_Edge(object):
    """
    Input feature class for baseline4
    Use InputExample_GNN like InputExample_GNN
    For [q, k, q1, q2, q3, k1, k2, k3]
    With max_length as 512
    <qk><qk1><qk2><qk3><kq1><kq2><kq3>
    Co-train with the BERT model part
    Futher cases:
    """
    def __init__(self,
                 input_ids_qk, input_mask_qk, segment_ids_qk,
                 input_ids_mix, input_mask_mix, segment_ids_mix,
                 mix_edge_cnt=7
                 ):
        self.input_ids_qk = input_ids_qk
        self.input_mask_qk = input_mask_qk
        self.segment_ids_qk = segment_ids_qk

        self.input_ids_mix = input_ids_mix
        self.input_mask_mix = input_mask_mix
        self.segment_ids_mix = segment_ids_mix
        self.split_index_mix = [0]*mix_edge_cnt     # the end_index for each edge, for further torch.split() with edge conv

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    """Base class for data converters for data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "QKVal.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "QKTest.tsv")), "test")


    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        """Some tensorflow_datasets datasets are not formatted the same way the GLUE datasets are.
        This method converts examples to the correct format."""
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class QKGNNRegProcessor(DataProcessor):
    """
    Processor fot the Q-K graph dataset
    with [id,label,query,doc,k1,k2,k3,q1,q2,q3,task_id]
    """
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # For further version
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("Read train file.")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "QKTrain.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "QKVal.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "QKTest.tsv")), "test")

    def get_labels(self):
        """Regression version"""
        return [None]

    def get_tasks(self):
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """
        Creates examples for the training and dev sets.
        For schema :id,label,query,doc,k1,k2,k3,q1,q2,q3,task_id
        Process to the two-class version : 0 and non-0
            For train data : mapping way is : 0，1，2，3, 0 is 0，else are 1
            For dev data : 0 and 1
        For line : delimiter is \t
        """
        examples = []
        for (i, line) in tqdm(enumerate(lines)):
            # donot escape the first line
            guid = line[0]
            text_a = line[2]
            text_b = line[3]
            label = "0" if line[1] == "0" else "1"
            k1 = line[4]
            k2 = line[5]
            k3 = line[6]
            q1 = line[7]
            q2 = line[8]
            q3 = line[9]
            task_id = line[10]
            # generate the examples
            examples.append(InputExample_GNN(
                guid=guid, text_a=text_a, text_b=text_b, label=label,
                k1=k1, k2=k2, k3=k3, q1=q1, q2=q2, q3=q3,  task=task_id
            ))
        return examples


class QKGNNClfProcessor(DataProcessor):
    """
    Processor fot the Q-K graph dataset
    with [id,label,query,doc,k1,k2,k3,q1,q2,q3,task_id]
    """
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        # For further version
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("Read train file.")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "QKTrain.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "QKVal.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "QKTest.tsv")), "test")

    def get_labels(self):
        """Classification version"""
        return ["0", "1"]

    def get_tasks(self):
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """
        Creates examples for the training and dev sets.
        For schema :id,label,query,doc,k1,k2,k3,q1,q2,q3,task_id
        Process to the two-class version : 0 and non-0
            For train data : mapping way is : 0，1，2，3, 0 is 0，else are 1
            For dev data : 0 and 1
        For line : delimiter is \t
        """
        examples = []
        for (i, line) in tqdm(enumerate(lines)):
            # donot escape the first line
            guid = line[0]
            text_a = line[2]
            text_b = line[3]
            label = "0" if line[1] == "0" else "1"
            k1 = line[4]
            k2 = line[5]
            k3 = line[6]
            q1 = line[7]
            q2 = line[8]
            q3 = line[9]
            task_id = line[10]
            # generate the examples
            examples.append(InputExample_GNN(
                guid=guid, text_a=text_a, text_b=text_b, label=label,
                k1=k1, k2=k2, k3=k3, q1=q1, q2=q2, q3=q3,  task=task_id
            ))
        return examples




def qkgnn_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = gnn_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = gnn_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    label_map = {label: i for i, label in enumerate(label_list)}


    # TODO : optimize this part if out of memory error occurs
    def convert_inputs_to_processed(inputs):
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        return input_ids, attention_mask, token_type_ids

    # TODO : optimize this part if out of memory error occurs
    def qkgnn_convert_single_exampl_to_feature(example, ex_index=10):
        inputs_q = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_k = tokenizer.encode_plus(example.text_b, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_qk = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_qk1 = tokenizer.encode_plus(example.text_a, example.k1, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_qk2 = tokenizer.encode_plus(example.text_a, example.k2, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_qk3 = tokenizer.encode_plus(example.text_a, example.k3, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_kq1 = tokenizer.encode_plus(example.text_b, example.q1, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_kq2 = tokenizer.encode_plus(example.text_b, example.q2, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_kq3 = tokenizer.encode_plus(example.text_b, example.q3, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_kk1 = tokenizer.encode_plus(example.text_b, example.k1, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_kk2 = tokenizer.encode_plus(example.text_b, example.k2, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_kk3 = tokenizer.encode_plus(example.text_b, example.k3, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_qq1 = tokenizer.encode_plus(example.text_a, example.q1, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_qq2 = tokenizer.encode_plus(example.text_a, example.q2, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_qq3 = tokenizer.encode_plus(example.text_a, example.q3, add_special_tokens=True, max_length=max_length, truncation=True)
        # generate [input_ids, input_mask, segment_ids]
        # for q self
        input_ids_q, input_mask_q, segment_ids_q = convert_inputs_to_processed(inputs_q)
        # for k self
        input_ids_k, input_mask_k, segment_ids_k = convert_inputs_to_processed(inputs_k)
        # for qk
        input_ids_qk, input_mask_qk, segment_ids_qk = convert_inputs_to_processed(inputs_qk)
        # for qk1
        input_ids_qk1, input_mask_qk1, segment_ids_qk1 = convert_inputs_to_processed(inputs_qk1)
        # for qk2
        input_ids_qk2, input_mask_qk2, segment_ids_qk2 = convert_inputs_to_processed(inputs_qk2)
        # for qk3
        input_ids_qk3, input_mask_qk3, segment_ids_qk3 = convert_inputs_to_processed(inputs_qk3)
        # for kq1
        input_ids_kq1, input_mask_kq1, segment_ids_kq1 = convert_inputs_to_processed(inputs_kq1)
        # for kq2
        input_ids_kq2, input_mask_kq2, segment_ids_kq2 = convert_inputs_to_processed(inputs_kq2)
        # for kq3
        input_ids_kq3, input_mask_kq3, segment_ids_kq3 = convert_inputs_to_processed(inputs_kq3)
        # for kk1
        input_ids_kk1, input_mask_kk1, segment_ids_kk1 = convert_inputs_to_processed(inputs_kk1)
        # for kk2
        input_ids_kk2, input_mask_kk2, segment_ids_kk2 = convert_inputs_to_processed(inputs_kk2)
        # for kk3
        input_ids_kk3, input_mask_kk3, segment_ids_kk3 = convert_inputs_to_processed(inputs_kk3)
        # for qq1
        input_ids_qq1, input_mask_qq1, segment_ids_qq1 = convert_inputs_to_processed(inputs_qq1)
        # for qq2
        input_ids_qq2, input_mask_qq2, segment_ids_qq2 = convert_inputs_to_processed(inputs_qq2)
        # for qq3
        input_ids_qq3, input_mask_qq3, segment_ids_qq3 = convert_inputs_to_processed(inputs_qq3)

        # generate label
        if output_mode == "classification" or output_mode == "classification2":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # log info
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("text_a: %s" % (example.text_a))
            logger.info("text_b: %s" % (example.text_b))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids_qk]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in input_mask_qk]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in segment_ids_qk]))
            logger.info("label: %s (id = %d)" % (example.label, label))
        # generate the feature for single example
        feature = InputFeatures_GNN(
            input_ids_q=input_ids_q,
            input_mask_q=input_mask_q,
            segment_ids_q=segment_ids_q,

            input_ids_k=input_ids_k,
            input_mask_k=input_mask_k,
            segment_ids_k=segment_ids_k,

            input_ids_qk=input_ids_qk,
            input_mask_qk=input_mask_qk,
            segment_ids_qk=segment_ids_qk,

            input_ids_qk1=input_ids_qk1,
            input_mask_qk1=input_mask_qk1,
            segment_ids_qk1=segment_ids_qk1,
            input_ids_qk2=input_ids_qk2,
            input_mask_qk2=input_mask_qk2,
            segment_ids_qk2=segment_ids_qk2,
            input_ids_qk3=input_ids_qk3,
            input_mask_qk3=input_mask_qk3,
            segment_ids_qk3=segment_ids_qk3,

            input_ids_kq1=input_ids_kq1,
            input_mask_kq1=input_mask_kq1,
            segment_ids_kq1=segment_ids_kq1,
            input_ids_kq2=input_ids_kq2,
            input_mask_kq2=input_mask_kq2,
            segment_ids_kq2=segment_ids_kq2,
            input_ids_kq3=input_ids_kq3,
            input_mask_kq3=input_mask_kq3,
            segment_ids_kq3=segment_ids_kq3,

            input_ids_qq1=input_ids_qq1,
            input_mask_qq1=input_mask_qq1,
            segment_ids_qq1=segment_ids_qq1,
            input_ids_qq2=input_ids_qq2,
            input_mask_qq2=input_mask_qq2,
            segment_ids_qq2=segment_ids_qq2,
            input_ids_qq3=input_ids_qq3,
            input_mask_qq3=input_mask_qq3,
            segment_ids_qq3=segment_ids_qq3,

            input_ids_kk1=input_ids_kk1,
            input_mask_kk1=input_mask_kk1,
            segment_ids_kk1=segment_ids_kk1,
            input_ids_kk2=input_ids_kk2,
            input_mask_kk2=input_mask_kk2,
            segment_ids_kk2=segment_ids_kk2,
            input_ids_kk3=input_ids_kk3,
            input_mask_kk3=input_mask_kk3,
            segment_ids_kk3=segment_ids_kk3,

            row_id=int(example.guid),
            label_id=label,
            task_id=task,
            is_real_example=True)
        return feature

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        features.append(qkgnn_convert_single_exampl_to_feature(example, ex_index=ex_index))

    return features



class InputFeatures_Node_GNN(object):
    """
    Input Feature class for baseline3
    A single set of features of data.
    Processed by tokenizer
        contains input_ids, attention_mask, token_type_ids
        - q
        - k
        - qk
        - k1
        - k2
        - k3
        - q1
        - q2
        - q3
    Other data:
        - row id: unique is for the feature
        - label_id: label(int or float)
        - task_id: for further usage
    """
    def __init__(self,
               input_ids_q,input_mask_q,segment_ids_q,
               input_ids_k,input_mask_k,segment_ids_k,
               input_ids_qk,input_mask_qk,segment_ids_qk,
               input_ids_k1,input_mask_k1,segment_ids_k1,
               input_ids_k2,input_mask_k2,segment_ids_k2,
               input_ids_k3,input_mask_k3,segment_ids_k3,
               input_ids_q1,input_mask_q1,segment_ids_q1,
               input_ids_q2,input_mask_q2,segment_ids_q2,
               input_ids_q3,input_mask_q3,segment_ids_q3,

               row_id,label_id,task_id,
               is_real_example=True):
        # frocessed for the edge sequence ids
        self.input_ids_q = input_ids_q
        self.input_mask_q = input_mask_q
        self.segment_ids_q = segment_ids_q

        self.input_ids_k = input_ids_k
        self.input_mask_k = input_mask_k
        self.segment_ids_k = segment_ids_k

        self.input_ids_qk = input_ids_qk
        self.input_mask_qk = input_mask_qk
        self.segment_ids_qk = segment_ids_qk

        self.input_ids_k1 = input_ids_k1
        self.input_mask_k1 = input_mask_k1
        self.segment_ids_k1 = segment_ids_k1

        self.input_ids_k2 = input_ids_k2
        self.input_mask_k2 = input_mask_k2
        self.segment_ids_k2 = segment_ids_k2

        self.input_ids_k3 = input_ids_k3
        self.input_mask_k3 = input_mask_k3
        self.segment_ids_k3 = segment_ids_k3

        self.input_ids_q1 = input_ids_q1
        self.input_mask_q1 = input_mask_q1
        self.segment_ids_q1 = segment_ids_q1

        self.input_ids_q2 = input_ids_q2
        self.input_mask_q2 = input_mask_q2
        self.segment_ids_q2 = segment_ids_q2

        self.input_ids_q3 = input_ids_q3
        self.input_mask_q3 = input_mask_q3
        self.segment_ids_q3 = segment_ids_q3



        self.row_id = row_id
        self.label_id = label_id
        self.is_real_example = is_real_example
        self.task_id = task_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def qkgnn_convert_examples_to_node_level_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = gnn_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = gnn_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    label_map = {label: i for i, label in enumerate(label_list)}


    # TODO : optimize this part if out of memory error occurs
    def convert_inputs_to_processed(inputs):
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        return input_ids, attention_mask, token_type_ids

    # TODO : optimize this part if out of memory error occurs
    def qkgnn_convert_single_exampl_to_feature(example, ex_index=10):
        inputs_q = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_k = tokenizer.encode_plus(example.text_b, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_qk = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_k1 = tokenizer.encode_plus(example.k1, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_k2 = tokenizer.encode_plus(example.k2, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_k3 = tokenizer.encode_plus(example.k3, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_q1 = tokenizer.encode_plus(example.q1, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_q2 = tokenizer.encode_plus(example.q2, add_special_tokens=True, max_length=max_length, truncation=True)
        inputs_q3 = tokenizer.encode_plus(example.q3, add_special_tokens=True, max_length=max_length, truncation=True)

        # generate [input_ids, input_mask, segment_ids]
        # for q self
        input_ids_q, input_mask_q, segment_ids_q = convert_inputs_to_processed(inputs_q)
        # for k self
        input_ids_k, input_mask_k, segment_ids_k = convert_inputs_to_processed(inputs_k)
        # for qk
        input_ids_qk, input_mask_qk, segment_ids_qk = convert_inputs_to_processed(inputs_qk)
        # for k1
        input_ids_k1, input_mask_k1, segment_ids_k1 = convert_inputs_to_processed(inputs_k1)
        # for k2
        input_ids_k2, input_mask_k2, segment_ids_k2 = convert_inputs_to_processed(inputs_k2)
        # for k3
        input_ids_k3, input_mask_k3, segment_ids_k3 = convert_inputs_to_processed(inputs_k3)
        # for q1
        input_ids_q1, input_mask_q1, segment_ids_q1 = convert_inputs_to_processed(inputs_q1)
        # for q2
        input_ids_q2, input_mask_q2, segment_ids_q2 = convert_inputs_to_processed(inputs_q2)
        # for q3
        input_ids_q3, input_mask_q3, segment_ids_q3 = convert_inputs_to_processed(inputs_q3)


        # generate label
        if output_mode == "classification" or output_mode == "classification2":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # log info
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("text_a: %s" % (example.text_a))
            logger.info("text_b: %s" % (example.text_b))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids_qk]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in input_mask_qk]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in segment_ids_qk]))
            logger.info("label: %s (id = %d)" % (example.label, label))
        # generate the feature for single example
        feature = InputFeatures_Node_GNN(
            input_ids_q=input_ids_q,
            input_mask_q=input_mask_q,
            segment_ids_q=segment_ids_q,

            input_ids_k=input_ids_k,
            input_mask_k=input_mask_k,
            segment_ids_k=segment_ids_k,

            input_ids_qk=input_ids_qk,
            input_mask_qk=input_mask_qk,
            segment_ids_qk=segment_ids_qk,

            input_ids_k1=input_ids_k1,
            input_mask_k1=input_mask_k1,
            segment_ids_k1=segment_ids_k1,
            input_ids_k2=input_ids_k2,
            input_mask_k2=input_mask_k2,
            segment_ids_k2=segment_ids_k2,
            input_ids_k3=input_ids_k3,
            input_mask_k3=input_mask_k3,
            segment_ids_k3=segment_ids_k3,

            input_ids_q1=input_ids_q1,
            input_mask_q1=input_mask_q1,
            segment_ids_q1=segment_ids_q1,
            input_ids_q2=input_ids_q2,
            input_mask_q2=input_mask_q2,
            segment_ids_q2=segment_ids_q2,
            input_ids_q3=input_ids_q3,
            input_mask_q3=input_mask_q3,
            segment_ids_q3=segment_ids_q3,



            row_id=int(example.guid),
            label_id=label,
            task_id=task,
            is_real_example=True)
        return feature

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        features.append(qkgnn_convert_single_exampl_to_feature(example, ex_index=ex_index))

    return features



gnn_tasks_num_labels = {
    "qk33reg": 1,
    "qk33clf": 2,
    "qk33regnode" : 1,
    "qk33clfnode" : 2
}

gnn_processors = {
    "qk33reg": QKGNNRegProcessor,
    "qk33clf": QKGNNClfProcessor,
    "qk33regnode" : QKGNNRegProcessor,
    "qk33clfnode" : QKGNNClfProcessor
}

gnn_output_modes = {
    "qk33reg": "regression",
    "qk33clf": "classification2",
    "qk33regnode": "regression",
    "qk33clfnode": "classification2",
}

if __name__ == '__main__':
    demo_example = InputExample_GNN(guid='12', text_a='2')
    print(demo_example.__repr__())