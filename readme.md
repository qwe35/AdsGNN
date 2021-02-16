# EdgeBert: A Click-Graph Enhanced Approach for Search Ads Matching

for WWW 2021

Improve data quality by semi-supervised gnn model

### Quick start
models
- Bert & MetaBert
    - add graph meta information into bert input 
    - use huggingface, see more in [huggingface](https://github.com/huggingface/transformers)
- EdgeBert
    - edge-level gnn model
        - bert as encoder to generate <Q, K> pairs feature
    - attention-based gnn model
    - training acceleration mechanism with associate bert encoder
- TokenNet
    - token-level gnn model
    - with local attention and global attention
- NodeGnn
    - node-leve gnn model
        - bert as encoder to generate node feature for Q and K
    - include:
        - GAT
        - GraphSAGE
 
semi-supervised learning 
- relevance dataset
    - query, keyword, label
    - human-labeled, expensive, limited-amount
- click dataset
    - query, keyword, is_click
    - from log, low cost, huge-amount
    - build <Q, K> graph from search ads click dataset
  
uitls
- metrics: roc_auc
- data_processor 
- token_net_generator


### dataset 

generate dummy dataset (qk & qk33gnn)
    
    cd data
    python generate_dummy_data.py

qk dataset
- schema : [rid,label,query,keyword,taskid]
- joined by \t 
- rid : row id for each line
- label : 0 or 1
- query
- keyword
- taskid : 0, for future multi-task learning, not used in this repo
- /dummy_data/qk320/

qk33gnn dataset 
- schema : [rid,label,query,doc,k1,k2,k3,q1,q2,q3,task_id]
- joined by \t 
- rid : row id for each line
- label : 0 or 1
- query
- keyword
- taskid : 0, for future multi-task learning, not used in this repo
- k1, k2, k3 : three query's neighbors from click-graph
- q1, q2, q3 : three keyword's neighbors from click-graph
- /dummy_data/qk33gnn320/

### models 

#### Bert & MetaBert
use orignal bert BertForSequenceClassification from huggingface. 

update data processor for QK-relevance task, see more in ./data/processors.py 

main difference from orginal huggingface
- change data loading strategy to fit input data schema
- change data mode from (dev, train) to (dev, test, train)
- create classification2 mode for QK task
- add new training function
    - add inference mode
    - update evaluating mechanism

MetaBert : concat node and neighbor's texts as new text input

script 
    
    python run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name qkclf \
    --do_train \
    --do_eval \
    --do_inference \
    --do_lower_case \
    --data_dir ./dummy_data/qk320/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ./dummy_data/qk320/meta_bert-base-uncased/

#### EdgeBert

edge-level model
- three edge kinds (node pair)
    - target edge <Q, K>
    - first-order edge : use_qk
    - second-order edge : use_kkqq

- acceleration version of EdgeBert
    - two bert encoders 
        - qk_bert, neighbor_bert
        - freeze neighbor_bert, update parameters from qk_bert every update_freezed_bert_parameter_steps training steps
    - cross training mechanism
    - around 20h\epoch to 4h\epoch
    - lower gpu cost, especially for input examples with more neighbor-edges in Graph

script 

    python run_edge_bert.py \
    --model_type edgebert \
    --model_name_or_path bert-base-uncased \
    --task_name qk33clf \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ./dummy_data/qk33gnn320/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --tensorboard_dir ./dummy_data/qk33gnn320/runs_edge_bert/ \
    --output_dir ./dummy_data/qk33gnn320/qk33clf-edge_bert_batch_size-16-unfreezed-part-qk-update-frequent-3-steps/ \
    --use_kkqq True \
    --use_qk True \
    --debug True \
    --eval_all_checkpoints \
    --evaluate_every_epoch \
    --inference_neighbor_bert True \
    --is_freeze_bert False \
    --load_pretrained_bert_checkpoint \
    --update_freezed_bert_parameter \
    --gradient_accumulation_steps 1 \
    --update_freezed_bert_parameter_steps 3
    
to close acceleration mechanism 

    --eval_all_checkpoints \
    --evaluate_every_epoch \
    --inference_neighbor_bert False \
    --is_freeze_bert False \
    --load_pretrained_bert_checkpoint \
    --gradient_accumulation_steps 1
    
to change the gnn part to [GraphBert](https://arxiv.org/pdf/2001.05140.pdf)-style gnn decoder model, update --model_type from edgebert to graphbert


#### TokenNet

see more in ./token_net

###### Convert the QK data to token net
see more in ./token_net/convert_to_token_net.py

1. extract bert word embedding
2. generate the token net with networkx
3. merge different token-net from several sources
    - relevance token net
    - click token net
4. generate offline sampling training dataset for token-net model

###### train TokenNet model

TokenNet
- use two data sources: relevance & click
- attention-based gnn model
    - local attention with neighbor token
    - global attention on two networks

Or you could use other gnn model like GAT, GraphSAGE to generate new token embeddings

#### EdgeBert with TokenNet

update BertEncoder structure, for original bert embedding layer
- word_embeddings
- position_embeddings
- segment_embeddings
Add token net embeddings into embedding layer

script 

    python run_edge_bert.py \
    --model_type tokennetedgebert \
    --model_name_or_path bert-base-uncased \
    --task_name qk33clf \
    --do_train \
    --do_eval \
    --do_inference \
    --inference_all_checkpoints \
    --do_lower_case \
    --data_dir ./dummy_data/qk33gnn320/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --tensorboard_dir ./dummy_data/qk33gnn320/runs_mix/ \
    --output_dir ./dummy_data/qk33gnn320/qk33clf-mix_unfreezed-part-qk-update-frequent-batch_size-16-500-steps/ \
    --use_kkqq True \
    --use_qk True \
    --debug True \
    --eval_all_checkpoints \
    --evaluate_every_epoch \
    --inference_neighbor_bert True \
    --is_freeze_bert False \
    --load_pretrained_bert_checkpoint \
    --update_freezed_bert_parameter \
    --gradient_accumulation_steps 1 \
    --update_freezed_bert_parameter_steps 3 \
    --gnn_token_embedding_path ./dummy_data/token_net/embedding_0.tsv \
    --is_freeze_gnn_token_embedding False 

#### Node-level model

previous models are edge-level models, we also accomplish some node-level gnn model demos for this task
- GAT
- GraphSAGE

- is_use_gnn: default True
    - True: use gnn part
    - False: do not use gnn part


script 

    python run_node_model.py \
    --model_type gat \
    --model_name_or_path bert-base-uncased \
    --task_name qk33clfnode \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ./dummy_data/qk33gnn320/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ./dummy_data/qk33gnn320/node_level/gat/ \
    --debug True \
    --eval_all_checkpoints \
    --gnn_aggregator meanpool \
    --gnn_head_nums 4 \
    --gnn_residual add
    
    python run_node_model.py \
    --model_type graphsage \
    --model_name_or_path bert-base-uncased \
    --task_name qk33clfnode \
    --do_train \
    --do_inference \
    --do_lower_case \
    --data_dir ./dummy_data/qk33gnn320/ \
    --max_seq_length 128 \
    --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --output_dir ./dummy_data/qk33gnn320/node_level/graphsage/ \
    --debug True \
    --eval_all_checkpoints \
    --gnn_aggregator meanpool \
    --gnn_head_nums 2 \
    --gnn_residual add
    
### Experiment Environment

    transformers=3.0.2
    python=3.6
    pyotrch=1.4.0 
    networkx