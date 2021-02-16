# -*- coding: UTF-8 -*-

"""
demo for run
"""

import argparse
import glob
import json
import logging
import os
import random
import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import get_linear_schedule_with_warmup
from .model import UnsupervisedTokenNet, TokenNetConfig
from .dataset import convert_examples_to_features, convert_to_Tensors, generate_model_inputs

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, mode_str='dev', task='token_net_pretrain'):
    assert mode_str in ["val", 'eval', 'train']
    # Load data features from cache or dataset file
    # use the default format
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            mode_str,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        features = convert_examples_to_features(args.data_dir)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and mode_str=='train':
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    dataset = convert_to_Tensors(features=features)

    return dataset




def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # other parameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=30000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    # for tensorboard setting
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="runs",
        help="tensorboard_dir"
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")


    # for model setting
    """
        feature_dim:int=768,
        head_nums:int=12,
        use_self_attn=False,
        attn_use_residual=True,
        attn_layer_num=4,
        attn_layer_drop_out_rate=0.2,
        attn_layer_head_num=4,
        feature_data_path=None
    """
    parser.add_argument("--use_self_attn", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None], default=False, help="use_self_attn for token-net model")
    parser.add_argument("--attn_use_residual", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None], default=True, help="attn_use_residual for token-net model")



    parser.add_argument("--feature_dim", type=int, default=768, help="feature_dim")
    parser.add_argument("--head_nums", type=int, default=4, help="head_nums for attention layer")
    parser.add_argument("--attn_layer_num", type=int, default=4, help="attn_layer_num for attention layer")
    parser.add_argument("--attn_layer_head_num", type=int, default=4, help="attn_layer_head_num for attention layer")
    parser.add_argument("--attn_layer_drop_out_rate", type=float, default=0.2, help="attn_layer_drop_out_rate for attention layer")

    parser.add_argument("--feature_data_path", type=str, default=None,
                        required=True,
                        help="feature_data_path for attention layer")

    args = parser.parse_args()


    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # setting tensorboardX

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.tensorboard_dir)
        tb_writer.add_text('args', str(args), 0)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    def get_current_model_config_from_args(current_args):
        """
            feature_dim:int=768,
            head_nums:int=12,
            use_self_attn=False,
            attn_use_residual=True,
            attn_layer_num=4,
            attn_layer_drop_out_rate=0.2,
            attn_layer_head_num=4,
            feature_data_path=None
        """
        tmp_config = TokenNetConfig(
            feature_dim=current_args.feature_dim,
            head_nums=current_args.head_nums,
            use_self_attn=current_args.use_self_attn,
            attn_use_residual=current_args.attn_use_residual,
            attn_layer_num=current_args.attn_layer_num,
            attn_layer_drop_out_rate=current_args.attn_layer_drop_out_rate,
            attn_layer_head_num=current_args.attn_layer_head_num,
            feature_data_path=current_args.feature_data_path,
        )
        return tmp_config

    config = get_current_model_config_from_args(current_args=args)
    model = TokenNetConfig(config)

    #################################  prepare for model training ################################
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    #logger.info(model)
    logger.info(config)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, mode_str='train', task='token_net_pretrain')
        global_step, tr_loss = train(args, train_dataset, model, tb_writer=tb_writer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.local_rank in [-1, 0]:
        tb_writer.close()


def train(args, train_dataset, model, tb_writer):
    """ Train the model """


    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
        num_workers=32,
        pin_memory=True,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        logger.info("   Use fp16 training  ")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # load other baseline bert checkpoint for transfer learning
        if args.load_pretrained_bert_checkpoint:
            global_step = 0
        # set global_step to gobal_step of last saved checkpoint from model path
        else:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    train_epoch_index = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # inputs is the dict object
            inputs = generate_model_inputs(batch=batch)

            # generate the input for model

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


                if args.local_rank in [-1,
                                       0] and args.save_steps > 0 and global_step % args.update_freezed_bert_parameter_steps == 0:
                    # update freeze bert
                    if args.update_freezed_bert_parameter and args.inference_neighbor_bert:
                        logger.info(
                            "Update_freezed_bert_parameter for epoch : {}, global step : {}".format(train_epoch_index,
                                                                                                    global_step))
                        # for multi-gpu cases
                        if hasattr(model, "module"):
                            model.module.update_freezed_bert_parameter()
                        else:
                            model.update_freezed_bert_parameter()



            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        # save-checkpoint for each epoch
        logger.info("Saving model checkpoint at epoch : {}".format(train_epoch_index))
        output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(output_dir)
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.bin"))  # save the model state_dict file

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)
        # evaluate for each epoch
        logger.info("Evaluating for epoch : {}".format(train_epoch_index))

        train_epoch_index += 1

    return global_step, tr_loss / global_step


if __name__ == "__main__":
    main()