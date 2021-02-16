import logging
import glob
import os
from tqdm import tqdm
import traceback
from itertools import cycle
from functools import partial

import numpy as np
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

class GnnKdDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_path, repeat=True, max_seq_len=20, column_count=5, text_separator=',', **kwargs):
        super().__init__()
        logger.info(f"reading data from {data_path}")
        self.data_path_iter = cycle([data_path]) if repeat else [data_path]
        self.max_seq_len = max_seq_len
        self.column_count = column_count
        self.text_separator = text_separator

    def __iter__(self):
        for data_path in self.data_path_iter:
            logger.info(f'loading data from {data_path}...')
            with open(data_path, 'r', encoding='utf-8') as reader:
                for line in reader:
                    try:
                        result = self._process_line(line)
                        if result is not None:
                            yield result
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(f"Got exception: {e}, traceback: {tb}")
    
    def _process_line(self, line):
        items = line.rstrip('\r\n').split('\t')
        if len(items) < self.column_count:
            return None
        line_id = items[0]
        title = items[1]
        title_ids = items[2]
        src_emb_text = items[3]
        target_emb_text = items[4]

        input_id, attention_mask = self.tokenize(title_ids)
        src_emb = self.get_emb_tensor(src_emb_text)
        target_emb = self.get_emb_tensor(target_emb_text)
        data = [input_id, attention_mask, src_emb, target_emb, line_id]
        return data

    def get_emb_tensor(self, emb_text):
        if not emb_text:
            # dummy emb for empty input
            emb = [0.0]
        else:
            emb = [float(x) for x in emb_text.split(self.text_separator)]
        emb_np = np.array(emb)
        return torch.from_numpy(emb_np).float()

    def tokenize(self, ids):
        ids = ids.strip()
        if not ids:
            return self.get_zero_tensor(), self.get_zero_tensor()
        ids_tensor = torch.LongTensor([int(x) for x in ids.split(',')])
        input_id = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        input_id[:len(ids_tensor)] = ids_tensor
        attention_mask[:len(ids_tensor)] = torch.ones(len(ids_tensor), dtype=torch.long)
        return input_id, attention_mask

    def get_zero_tensor(self):
        return torch.zeros(self.max_seq_len, dtype=torch.long)

class GnnKdInferenceDataset(GnnKdDataset):
    def __init__(self, data_path, max_seq_len=64, column_count=3, text_separator=',', **kwargs):
        super().__init__(data_path, repeat=False, max_seq_len=max_seq_len, column_count=column_count, text_separator=text_separator)

    def _process_line(self, line):
        items = line.rstrip('\r\n').split('\t')
        if len(items) < self.column_count:
            return None
        line_id = items[0]
        title_ids = items[1]
        src_emb_text = items[2]

        input_id, attention_mask = self.tokenize(title_ids)
        src_emb = self.get_emb_tensor(src_emb_text)
        assert src_emb.size(0) == 128, f"assertion error: src_emb size is not 128: {src_emb}"
        data = [input_id, attention_mask, src_emb, line_id]
        return data