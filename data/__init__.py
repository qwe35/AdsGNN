# -*- coding: UTF-8 -*-
from .utils import DataProcessor, InputExample, InputFeatures, SingleSentenceClassificationProcessor
from .metrics import glue_compute_metrics
from .processors import (
    glue_convert_examples_to_features,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
)
from .processors_edge_bert import (
    qkgnn_convert_examples_to_features,
    gnn_output_modes,
    gnn_processors,
    gnn_tasks_num_labels,
    qkgnn_convert_examples_to_node_level_features
)
from .metrics import is_sklearn_available
if is_sklearn_available():
    from .metrics import glue_compute_metrics, xnli_compute_metrics