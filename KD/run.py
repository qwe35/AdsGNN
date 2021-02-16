import os
import logging

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)
logger = logging.getLogger(__name__)

def run(cmd):
    logger.info(f"run command: {cmd}")
    retval = os.system(cmd)
    if retval != 0:
        raise Exception(f"Finished cmd {cmd} with return value: {retval}")

def train_model_w_unseen():
    cmd = r"python main.py " \
        "--mode train " \
        "--data_path ./bert_kd_data_train.txt --valid_data_path ./bert_kd_data_valid_100k.txt " \
        "--transformer_pretrained_weights google/bert_uncased_L-2_H-128_A-2 " \
        "--num_data_workers 4 --pin_memory --use_src_emb " \
        "--model_dir bert_data_kd_model_with_src_emb --max_seq_len 64 " \
        "--max_steps 10000000 --warmup_steps 2000 " \
        "--learning_rate 5e-5 " \
        "--max_seq_len 64 --dense_sizes 1024,128 " \
        "--log_steps 10 --eval_steps 1000 --save_steps 2000 --batch_size 6000 "
    run(cmd)

def train_model_title_only():
    cmd = r"python main.py " \
        "--mode train " \
        "--data_path ./bert_kd_data_train.txt --valid_data_path ./bert_kd_data_valid_100k.txt " \
        "--transformer_pretrained_weights google/bert_uncased_L-2_H-128_A-2 " \
        "--num_data_workers 4 --pin_memory " \
        "--model_dir bert_data_kd_model_title_only --max_seq_len 64 " \
        "--max_steps 10000000 --warmup_steps 2000 " \
        "--learning_rate 5e-5 " \
        "--max_seq_len 64 --dense_sizes 1024,128 " \
        "--log_steps 10 --eval_steps 1000 --save_steps 2000 --batch_size 6000 "
    run(cmd)

def train_model_mlp():
    cmd = r"python main.py " \
        "--mode train " \
        "--data_path ./kd_data_train.txt --valid_data_path ./kd_data_valid_100k.txt " \
        "--num_data_workers 0 --pin_memory --use_src_emb " \
        "--model_dir kd_model_mlp_only --max_seq_len 64 " \
        "--max_steps 10000000 --warmup_steps 2000 " \
        "--learning_rate 5e-5 " \
        "--max_seq_len 64 --dense_sizes 1024,1024,50 " \
        "--log_steps 10 --eval_steps 1000 --save_steps 2000 --batch_size 8000 "
    run(cmd)

def run_inference():
    cmd = r"CUDA_VISIBLE_DEVICES= python main.py " \
        "--mode inference " \
        "--inference_input_path ./test_input.txt " \
        "--inference_output_path ./test_output.txt " \
        "--model_dir bert_data_kd_model_with_src_emb --transformer_pretrained_weights dummy " \
        "--max_seq_len 64 --log_steps 1000 "
        #" --use_src_emb " \
    run(cmd)

def main():
    #train_model_title_only()
    #train_model_w_unseen()
    #train_model_mlp()
    run_inference()

if __name__ == "__main__":
    main()