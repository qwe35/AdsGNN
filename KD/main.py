from argparse import ArgumentError
import os
import argparse
import logging
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
#import horovod.torch as hvd

from data_loader import GnnKdDataset, GnnKdInferenceDataset
from models import TransformerEncoder, MLP

from transformers import AdamW
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, required=True, help="train, evaluate, inference")
    parser.add_argument("--data_path", default=None, type=str, help="The input data path.")
    parser.add_argument("--model_dir", default=None, type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--valid_data_path", default=None, type=str, help="The validation data path.")
    parser.add_argument("--inference_input_path", default=None, type=str, help="Input file path for inference.")
    parser.add_argument("--inference_output_path", default=None, type=str, help="Output file path for inference.")
    parser.add_argument("--transformer_pretrained_weights", default=None, type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--max_seq_len", default=64, type=int, help="Maximum sequence length in terms of word count.")
    parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
    parser.add_argument("--num_data_workers", default=0, type=int, help="Dataloader num workers.")
    parser.add_argument("--pin_memory", action='store_true', help="Pin memory in data loader")
    parser.add_argument("--use_src_emb", action='store_true', help="Whether to use the source embedding tensor in the model.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning rate.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_steps", default=10000000, type=int, help="Stop training after N steps.")
    parser.add_argument("--dense_sizes", default="50", type=str, help="The dense layer sizes separated by ',', last one will be output dimension.")
    parser.add_argument("--log_steps", default=20, type=int, help="Log training status every N steps.")
    parser.add_argument('--logging_file', type=str, default=None, help='Log file to save.')
    parser.add_argument("--dropout_rate", default=0.2, type=float, help="Dropout rate.")
    parser.add_argument("--eval_steps", default=200, type=int, help="Do evaluation every N steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    args = parser.parse_args()
    return args

def tally_parameters(model):
    n_params = 0
    for name, param in model.named_parameters():
        n_params += param.nelement()
    return n_params

def get_data_loader(args, data_path, mode, device, drop_last=True):
    if data_path is None:
        return None, None
    repeat = (mode == 'train')
    dataset = GnnKdDataset(data_path, max_seq_len=args.max_seq_len, repeat=repeat, device=device)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=args.num_data_workers if (mode == 'train') else 0, pin_memory=args.pin_memory, drop_last=drop_last)
    return dataset, data_loader

def get_infer_data_loader(args, data_path, device):
    if data_path is None:
        raise ArgumentError("data path is None!")
    dataset = GnnKdInferenceDataset(data_path, max_seq_len=args.max_seq_len, device=device)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=None, num_workers=0, pin_memory=False, drop_last=False)
    return data_loader

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps)
    # Add Horovod Distributed Optimizer
    #optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    return optimizer, scheduler

def run_model(args, model, loss_funct, batch_data, device):
    input_ids, attention_mask, src_emb, target_emb, line_id = batch_data
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    src_emb = src_emb.to(device)
    target_emb = target_emb.to(device)

    if args.transformer_pretrained_weights:
        predict_emb = model(input_ids, attention_mask, src_emb)
    else:
        predict_emb = model(src_emb)
    loss = loss_funct(predict_emb, target_emb)
    return loss

def evaluate_impl(args, data_loader, model, loss_funct, tb_writer=None, global_step=-1, device='cpu'):
    with torch.no_grad():
        losses = []
        n_examples = 0
        for batch_data in data_loader:
            n_examples += len(batch_data[0])
            loss = run_model(args, model, loss_funct, batch_data, device)
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        logger.info('step: {}, val loss: {:.6f}, example count: {}'.format(global_step, avg_loss, n_examples))
        if tb_writer:
            tb_writer.add_scalar('valid_loss', avg_loss, global_step)
        return avg_loss

def save_model(args, model, tag=None):
    model_dir = args.model_dir if tag is None else os.path.join(args.model_dir, 'checkpoint-{}'.format(tag))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    torch.save(model_to_save, os.path.join(model_dir, 'pytorch_model.bin'))
    torch.save(args, os.path.join(model_dir, 'training_args.bin'))
    logger.info("Saving model checkpoint to %s", model_dir)

def train(args):
    # Initialize Horovod
    #hvd.init()
    if torch.cuda.is_available():
        # Pin GPU to be used to process local rank (one GPU per process)
        #torch.cuda.set_device(hvd.local_rank())
        #device = torch.device(f"cuda:{hvd.local_rank()}")
        device = torch.device(f"cuda:0")
    else:
        device = torch.device('cpu')

    #if hvd.local_rank() == 0:
    if True:
        if not args.logging_file:
            args.logging_file = os.path.join(args.model_dir, 'log.txt')
        if not os.path.exists(args.model_dir):
            os.mkdir(args.model_dir)
        else:
            path = os.path.join(args.model_dir, "events.out.tfevents.*")
            os.system(f"rm -rf {path}")
        if os.path.exists(args.logging_file):
            os.remove(args.logging_file)

        fh = logging.FileHandler(args.logging_file, 'w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    for param in dir(args):
        if not param.startswith("_"):
            logger.info(f"{param}={getattr(args, param)}")

    logger.info(f'using device: {device}')

    train_dataset, train_loader = get_data_loader(args, args.data_path, mode='train', device=device)
    valid_dataset, valid_loader = get_data_loader(args, args.valid_data_path, mode='eval', device=device, drop_last=True)

    tb_writer = SummaryWriter(args.model_dir)

    if args.transformer_pretrained_weights:
        model = TransformerEncoder(args.transformer_pretrained_weights, dense_sizes=args.dense_sizes, dropout_rate=args.dropout_rate, use_src_emb=args.use_src_emb)
    else:
        input_size = int(args.dense_sizes.split(',')[-1])
        model = MLP(input_size=input_size, dense_sizes=args.dense_sizes, dropout_rate=args.dropout_rate)
    model = model.to(device)
    logger.info(model)
    n_params = tally_parameters(model)
    logger.info(f"parameters count: {n_params}")
    if args.transformer_pretrained_weights:
        n_params_transformer = tally_parameters(model.transformer_model)
        logger.info(f"transformer parameters count: {n_params_transformer}")

    #loss_funct = torch.nn.MSELoss()
    loss_funct = torch.nn.L1Loss()
    optimizer, scheduler = get_optimizer(args, model)

    global_step = 0
    train_loss = []
    start_time = time.time()
    for batch_data in train_loader:
        if global_step == args.max_steps:
            break
        optimizer.zero_grad()

        loss = run_model(args, model, loss_funct, batch_data, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler:
            scheduler.step()  # Update learning rate schedule

        train_loss.append(loss.item())
        if global_step % args.log_steps == 0:
            avg_loss = np.mean(train_loss)
            logger.info('Train step {}: avg train loss: {:.6f} seconds: {:.3f}'.format(global_step, avg_loss, time.time()-start_time))
            if scheduler:
                tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
            tb_writer.add_scalar('train_loss', avg_loss, global_step)
            train_loss = []
            start_time = time.time()

        if global_step % args.eval_steps == 0:# and hvd.local_rank() == 0:
            model = model.eval()
            evaluate_impl(args, valid_loader, model, loss_funct, tb_writer, global_step, device)
            model = model.train()

        if global_step % args.save_steps == 0:# and hvd.local_rank() == 0:
            logger.info(f'saving model to {args.model_dir}')
            save_model(args, model)
        
        global_step += 1

def inference(args):
    model_path = os.path.join(args.model_dir, 'pytorch_model.bin')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = torch.load(model_path)
    else:
        device = torch.device("cpu")
        model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()
    infer_loader = get_infer_data_loader(args, args.inference_input_path, device=device)

    with torch.no_grad():
        with open(args.inference_output_path, 'w', encoding='utf-8') as writer:
            counter = 0
            prev_counter = counter
            start = time.time()
            for batch_data in infer_loader:
                input_ids, attention_mask, src_emb, line_id = batch_data
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                if args.use_src_emb:
                    src_emb = src_emb.to(device)
                predict_emb = model(input_ids, attention_mask, src_emb)

                assert len(line_id) == len(predict_emb), f"len(label) ({len(line_id)}) not equal to label(encodings) ({len(predict_emb)})"
                for label, encoding in zip(line_id, predict_emb):
                    encoding_str = ','.join(map(lambda x: '{:.6f}'.format(x), encoding))
                    writer.write(f'{label}\t{encoding_str}\n')
                counter += len(line_id)
                if (counter - prev_counter) >= args.log_steps:
                    end = time.time()
                    speed = (counter - prev_counter) / (end - start + 1e-8)
                    logger.info(f'finished: {counter}, speed: ' + '{:.2f}'.format(speed) + ' lines/sec')
                    prev_counter = counter
                    start = time.time()

def main():
    args = parse_arguments()
    if args.mode == 'train':
        train(args)
    elif args.mode == 'evaluate':
        raise NotImplementedError()
    elif args.mode == 'inference':
        inference(args)
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
