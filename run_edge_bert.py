# -*- coding: UTF-8 -*-
"""

use pretrained bert-based-uncased

"""

import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup
)

from data import glue_compute_metrics as compute_metrics
from data import qkgnn_convert_examples_to_features as convert_examples_to_features
from data import gnn_output_modes as output_modes
from data import gnn_processors as processors

from model import ModelQKEdgeBert, ModelQKEdgeBertConfig, ModelQKEdgeBERT, ModelQKEdgeBERTConfig
from model import TokenNetConfig, TokenNetBertModel
from model import ModelQKEdgeBertConfig, ModelQKEdgeBertWithTokenNet

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "edgebert": (ModelQKEdgeBertConfig, ModelQKEdgeBert, BertTokenizer),
    "graphbert": (ModelQKEdgeBERTConfig, ModelQKEdgeBERT, BertTokenizer),
    "tokennet": (TokenNetConfig, TokenNetBertModel, BertTokenizer),
    "tokennetedgebert": (ModelQKEdgeBertConfig, ModelQKEdgeBertWithTokenNet, BertTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def convert_to_Tensors(features, output_mode):
    input_ids_q = torch.tensor([f.input_ids_q for f in features], dtype=torch.long)
    input_mask_q = torch.tensor([f.input_mask_q for f in features], dtype=torch.long)
    segment_ids_q = torch.tensor([f.segment_ids_q for f in features], dtype=torch.long)

    input_ids_k = torch.tensor([f.input_ids_k for f in features], dtype=torch.long)
    input_mask_k = torch.tensor([f.input_mask_k for f in features], dtype=torch.long)
    segment_ids_k = torch.tensor([f.segment_ids_k for f in features], dtype=torch.long)

    input_ids_qk = torch.tensor([f.input_ids_qk for f in features], dtype=torch.long)
    input_mask_qk = torch.tensor([f.input_mask_qk for f in features], dtype=torch.long)
    segment_ids_qk = torch.tensor([f.segment_ids_qk for f in features], dtype=torch.long)

    input_ids_qk1 = torch.tensor([f.input_ids_qk1 for f in features], dtype=torch.long)
    input_mask_qk1 = torch.tensor([f.input_mask_qk1 for f in features], dtype=torch.long)
    segment_ids_qk1 = torch.tensor([f.segment_ids_qk1 for f in features], dtype=torch.long)

    input_ids_qk2 = torch.tensor([f.input_ids_qk2 for f in features], dtype=torch.long)
    input_mask_qk2 = torch.tensor([f.input_mask_qk2 for f in features], dtype=torch.long)
    segment_ids_qk2 = torch.tensor([f.segment_ids_qk2 for f in features], dtype=torch.long)

    input_ids_qk3 = torch.tensor([f.input_ids_qk3 for f in features], dtype=torch.long)
    input_mask_qk3 = torch.tensor([f.input_mask_qk3 for f in features], dtype=torch.long)
    segment_ids_qk3 = torch.tensor([f.segment_ids_qk3 for f in features], dtype=torch.long)

    input_ids_kq1 = torch.tensor([f.input_ids_kq1 for f in features], dtype=torch.long)
    input_mask_kq1 = torch.tensor([f.input_mask_kq1 for f in features], dtype=torch.long)
    segment_ids_kq1 = torch.tensor([f.segment_ids_kq1 for f in features], dtype=torch.long)

    input_ids_kq2 = torch.tensor([f.input_ids_kq2 for f in features], dtype=torch.long)
    input_mask_kq2 = torch.tensor([f.input_mask_kq2 for f in features], dtype=torch.long)
    segment_ids_kq2 = torch.tensor([f.segment_ids_kq2 for f in features], dtype=torch.long)

    input_ids_kq3 = torch.tensor([f.input_ids_kq3 for f in features], dtype=torch.long)
    input_mask_kq3 = torch.tensor([f.input_mask_kq3 for f in features], dtype=torch.long)
    segment_ids_kq3 = torch.tensor([f.segment_ids_kq3 for f in features], dtype=torch.long)

    input_ids_qq1 = torch.tensor([f.input_ids_qq1 for f in features], dtype=torch.long)
    input_mask_qq1 = torch.tensor([f.input_mask_qq1 for f in features], dtype=torch.long)
    segment_ids_qq1 = torch.tensor([f.segment_ids_qq1 for f in features], dtype=torch.long)

    input_ids_qq2 = torch.tensor([f.input_ids_qq2 for f in features], dtype=torch.long)
    input_mask_qq2 = torch.tensor([f.input_mask_qq2 for f in features], dtype=torch.long)
    segment_ids_qq2 = torch.tensor([f.segment_ids_qq2 for f in features], dtype=torch.long)

    input_ids_qq3 = torch.tensor([f.input_ids_qq3 for f in features], dtype=torch.long)
    input_mask_qq3 = torch.tensor([f.input_mask_qq3 for f in features], dtype=torch.long)
    segment_ids_qq3 = torch.tensor([f.segment_ids_qq3 for f in features], dtype=torch.long)

    input_ids_kk1 = torch.tensor([f.input_ids_kk1 for f in features], dtype=torch.long)
    input_mask_kk1 = torch.tensor([f.input_mask_kk1 for f in features], dtype=torch.long)
    segment_ids_kk1 = torch.tensor([f.segment_ids_kk1 for f in features], dtype=torch.long)

    input_ids_kk2 = torch.tensor([f.input_ids_kk2 for f in features], dtype=torch.long)
    input_mask_kk2 = torch.tensor([f.input_mask_kk2 for f in features], dtype=torch.long)
    segment_ids_kk2 = torch.tensor([f.segment_ids_kk2 for f in features], dtype=torch.long)

    input_ids_kk3 = torch.tensor([f.input_ids_kk3 for f in features], dtype=torch.long)
    input_mask_kk3 = torch.tensor([f.input_mask_kk3 for f in features], dtype=torch.long)
    segment_ids_kk3 = torch.tensor([f.segment_ids_kk3 for f in features], dtype=torch.long)

    if output_mode == "classification" or output_mode == "classification2":
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.float)

    row_id = torch.tensor([f.row_id for f in features], dtype=torch.long)

    # generate dataset

    dataset = TensorDataset(
        input_ids_q, input_mask_q, segment_ids_q,
        input_ids_k, input_mask_k, segment_ids_k,
        input_ids_qk, input_mask_qk, segment_ids_qk,
        input_ids_qk1, input_mask_qk1, segment_ids_qk1,
        input_ids_qk2, input_mask_qk2, segment_ids_qk2,
        input_ids_qk3, input_mask_qk3, segment_ids_qk3,
        input_ids_kq1, input_mask_kq1, segment_ids_kq1,
        input_ids_kq2, input_mask_kq2, segment_ids_kq2,
        input_ids_kq3, input_mask_kq3, segment_ids_kq3,
        input_ids_qq1, input_mask_qq1, segment_ids_qq1,
        input_ids_qq2, input_mask_qq2, segment_ids_qq2,
        input_ids_qq3, input_mask_qq3, segment_ids_qq3,
        input_ids_kk1, input_mask_kk1, segment_ids_kk1,
        input_ids_kk2, input_mask_kk2, segment_ids_kk2,
        input_ids_kk3, input_mask_kk3, segment_ids_kk3,
        row_id, all_labels)

    return dataset


def generate_model_inputs(batch, FLAGS=None):
    features = {
        "input_ids_q": batch[0], "input_mask_q": batch[1], "segment_ids_q": batch[2],
        "input_ids_k": batch[3], "input_mask_k": batch[4], "segment_ids_k": batch[5],
        "input_ids_qk": batch[6], "input_mask_qk": batch[7], "segment_ids_qk": batch[8],
        "input_ids_qk1": batch[9], "input_mask_qk1": batch[10], "segment_ids_qk1": batch[11],
        "input_ids_qk2": batch[12], "input_mask_qk2": batch[13], "segment_ids_qk2": batch[14],
        "input_ids_qk3": batch[15], "input_mask_qk3": batch[16], "segment_ids_qk3": batch[17],
        "input_ids_kq1": batch[18], "input_mask_kq1": batch[19], "segment_ids_kq1": batch[20],
        "input_ids_kq2": batch[21], "input_mask_kq2": batch[22], "segment_ids_kq2": batch[23],
        "input_ids_kq3": batch[24], "input_mask_kq3": batch[25], "segment_ids_kq3": batch[26],
        "input_ids_qq1": batch[27], "input_mask_qq1": batch[28], "segment_ids_qq1": batch[29],
        "input_ids_qq2": batch[30], "input_mask_qq2": batch[31], "segment_ids_qq2": batch[32],
        "input_ids_qq3": batch[33], "input_mask_qq3": batch[34], "segment_ids_qq3": batch[35],
        "input_ids_kk1": batch[36], "input_mask_kk1": batch[37], "segment_ids_kk1": batch[38],
        "input_ids_kk2": batch[39], "input_mask_kk2": batch[40], "segment_ids_kk2": batch[41],
        "input_ids_kk3": batch[42], "input_mask_kk3": batch[43], "segment_ids_kk3": batch[44],
        "row_id": batch[-2], "label_id": batch[-1]
    }
    input_ids = features["input_ids_qk"]
    input_mask = features["input_mask_qk"]
    segment_ids = features["segment_ids_qk"]

    # input_rowid = features["row_id"]
    # input_taskid = features["task_id"]

    # gnn added features
    qself = [features["input_ids_q"], features["input_mask_q"], features["segment_ids_q"]]
    kself = [features["input_ids_k"], features["input_mask_k"], features["segment_ids_k"]]
    qks = [
        features["input_ids_qk1"], features["input_mask_qk1"], features["segment_ids_qk1"],
        features["input_ids_qk2"], features["input_mask_qk2"], features["segment_ids_qk2"],
        features["input_ids_qk3"], features["input_mask_qk3"], features["segment_ids_qk3"]]
    kqs = [
        features["input_ids_kq1"], features["input_mask_kq1"], features["segment_ids_kq1"],
        features["input_ids_kq2"], features["input_mask_kq2"], features["segment_ids_kq2"],
        features["input_ids_kq3"], features["input_mask_kq3"], features["segment_ids_kq3"]]
    qqs = [
        features["input_ids_qq1"], features["input_mask_qq1"], features["segment_ids_qq1"],
        features["input_ids_qq2"], features["input_mask_qq2"], features["segment_ids_qq2"],
        features["input_ids_qq3"], features["input_mask_qq3"], features["segment_ids_qq3"]]
    kks = [
        features["input_ids_kk1"], features["input_mask_kk1"], features["segment_ids_kk1"],
        features["input_ids_kk2"], features["input_mask_kk2"], features["segment_ids_kk2"],
        features["input_ids_kk3"], features["input_mask_kk3"], features["segment_ids_kk3"]]
    # input_ids_q = features["input_ids_q"]

    if FLAGS.use_kkqq and FLAGS.use_qk:
        namelist = ["qself", "kself", "qk1", "qk2", "qk3", "kq1", "kq2", "kq3", "qq1", "qq2", "qq3", "kk1", "kk2",
                    "kk3"]
        nei_models_num = 14
        input_neighbor = qself + kself + qks + kqs + qqs + kks
    elif not FLAGS.use_kkqq and FLAGS.use_qk:
        namelist = ["qself", "kself", "qk1", "qk2", "qk3", "kq1", "kq2", "kq3"]
        nei_models_num = 8
        input_neighbor = qself + kself + qks + kqs
    elif FLAGS.use_kkqq and not FLAGS.use_qk:
        namelist = ["qself", "kself", "qq1", "qq2", "qq3", "kk1", "kk2", "kk3"]
        nei_models_num = 8
        input_neighbor = qself + kself + qqs + kks
    else:
        raise KeyError

    label_id = features['label_id']

    row_id = features['row_id']

    output_ret = {
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids,
        'label_id': label_id,
        'row_id': row_id,
        'input_neighbor': input_neighbor,
        'namelist': namelist
    }

    return output_ret


def train(args, train_dataset, model, tokenizer, tb_writer):
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

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
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
            inputs = generate_model_inputs(batch=batch, FLAGS=args)

            """
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids,

            """
            if args.model_type == "tokennet":
                inputs['attention_mask'] = inputs['input_mask']
                inputs['token_type_ids'] = inputs['segment_ids']
                inputs['labels'] = inputs['label_id']
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

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, data_mode="dev")
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))
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

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    # model_to_save.save_pretrained(output_dir)
                    torch.save(model_to_save.state_dict(),
                               args.output_dir + '/model.bin')  # save the model state_dict file
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

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
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)
        # evaluate for each epoch
        logger.info("Evaluating for epoch : {}".format(train_epoch_index))

        if (
                args.local_rank == -1 and args.evaluate_every_epoch
        ):  # Only evaluate when single GPU otherwise metrics may not average well
            epcoh_logs = {}
            results = evaluate(args, model, tokenizer, data_mode="dev")
            tb_writer.add_text('evaluate results epoch', str(results), train_epoch_index)
            for key, value in results.items():
                eval_key = "epoch_eval_{}".format(key)
                epcoh_logs[eval_key] = value
            for key, value in epcoh_logs.items():
                tb_writer.add_scalar(key, value, train_epoch_index)

        # update freeze bert
        if args.update_freezed_bert_parameter and args.inference_neighbor_bert:
            logger.info("Update_freezed_bert_parameter for epoch : {}".format(train_epoch_index))
            # for multi-gpu cases
            if hasattr(model, "module"):
                model.module.update_freezed_bert_parameter()
            else:
                model.update_freezed_bert_parameter()

        train_epoch_index += 1

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", data_mode="test"):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, data_mode=data_mode)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = generate_model_inputs(batch=batch, FLAGS=args)

                if args.model_type == "tokennet":
                    inputs['attention_mask'] = inputs['input_mask']
                    inputs['token_type_ids'] = inputs['segment_ids']
                    inputs['labels'] = inputs['label_id']

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["label_id"].detach().cpu().numpy()
                if args.debug:
                    print(preds)

            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_id"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        elif args.output_mode == "classification2":
            # for specific qk clf roc_auc task
            # preds [eval_cnt, 2]
            preds = torch.softmax(torch.tensor(preds, dtype=torch.float64), dim=1)
            preds = preds[:, 1].numpy()  # target score for roc_auc
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        output_eval_file = eval_output_dir + '/' + prefix + '/' + "eval_results.txt"
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def inference(args, model, tokenizer, output_folder=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (output_folder, output_folder + "-MM") if args.task_name == "mnli" else (output_folder,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        # so add row_id to ensure model output
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # TODO support DataParallel by add row_id into inputs
        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Inference
        logger.info("***** Running evaluation {} *****".format(output_folder))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = generate_model_inputs(batch=batch, FLAGS=args)

                if args.model_type == "tokennet":
                    inputs['attention_mask'] = inputs['input_mask']
                    inputs['token_type_ids'] = inputs['segment_ids']
                    inputs['labels'] = inputs['label_id']

                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["label_id"].detach().cpu().numpy()
                if "row_id" in inputs:
                    input_row_ids = inputs["row_id"].detach().cpu().numpy()
                else:
                    input_row_ids = None
                    assert (args.n_gpu <= 1), "inference for multi-gpu without input row_id is dangerous"
                if args.debug:
                    print(preds)

            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["label_id"].detach().cpu().numpy(), axis=0)
                if "row_id" in inputs:
                    input_row_ids = np.append(input_row_ids, inputs["row_id"].detach().cpu().numpy(), axis=0)
                else:
                    input_row_ids = None
                    assert (args.n_gpu <= 1), "inference for multi-gpu without input row_id is dangerous"

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        elif args.output_mode == "classification2":
            preds = np.squeeze(preds)
            # for specific qk clf roc_auc task
            # # preds [eval_cnt, 2]
            # preds = torch.softmax(torch.tensor(preds, dtype=torch.float64), dim=1)
            # preds = preds[:,1].numpy() # target score for roc_auc

        output_inference_file = eval_output_dir + "/" + "eval_inferece_results"
        np.savetxt(fname=output_inference_file + "_preds.tsv", X=preds, delimiter='\t')
        np.savetxt(fname=output_inference_file + "_origin_label.tsv", X=out_label_ids, delimiter='\t')
        if input_row_ids is not None:
            np.savetxt(fname=output_inference_file + "_input_row_ids.tsv", X=input_row_ids, delimiter='\t')
        preds = np.expand_dims(preds, axis=-1) if preds.ndim == 1 else preds
        out_label_ids = np.expand_dims(out_label_ids, axis=-1)
        if input_row_ids is None:
            ret_data = np.concatenate([out_label_ids, preds], axis=-1)
            np.savetxt(fname=output_inference_file + "_inference_result_without_row_id.tsv", X=ret_data, delimiter='\t')
        else:
            input_row_ids = np.expand_dims(input_row_ids, axis=-1)
            ret_data = np.concatenate([input_row_ids, out_label_ids, preds], axis=-1)
            np.savetxt(fname=output_inference_file + "_inference_result_with_row_id.tsv", X=ret_data, delimiter='\t')
            with open(output_inference_file + "_inference_result_with_row_id.npy", "wb") as f:
                np.save(f, ret_data)

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, data_mode=None):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if data_mode is None:
        if evaluate:
            data_mode = "test"
        else:
            data_mode = "train"
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if args.processed_cached_features_file_path is not None and len(args.processed_cached_features_file_path) > 0:
        cached_features_file = args.processed_cached_features_file_path
    else:
        # use the default format
        cached_features_file = os.path.join(
            args.data_dir,
            "cached_{}_{}_{}_{}".format(
                data_mode,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if evaluate:
            if data_mode == "dev":
                examples = (processor.get_dev_examples(args.data_dir))
            else:
                examples = (processor.get_test_examples(args.data_dir))
        else:
            examples = (processor.get_train_examples(args.data_dir))
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    # TODO update this part for dataset then
    # for there are more

    dataset = convert_to_Tensors(features=features, output_mode=output_mode)

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
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--task_name",
        default="qk33reg",
        type=str,
        required=True,
        help="The name of the task to train selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_inference", action="store_true", help="Whether to run inference on the dev dataset")
    parser.add_argument(
        "--inference_all_checkpoints", action="store_true", help="inference_all_checkpoints for model dir",
    )
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

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
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=30000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

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

    # gnn related edge conv model part
    parser.add_argument("--gat_acts", type=str, default="leaky_relu", help="graphsage activations every layer")
    parser.add_argument("--use_residual", type=bool, default=False, help="Whether use skip connection.")
    parser.add_argument("--aggregator", type=str, default="mean",
                        help="Sage aggregator: gcn, mean, meanpool, maxpool, attention")
    parser.add_argument("--use_kkqq", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None], default=True,
                        help="Whether use qq pairs and kk pairs.")
    parser.add_argument("--use_qk", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None], default=True,
                        help="Whether use qk pairs and kq pairs.")
    parser.add_argument("--use_node", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None], default=True,
                        help="Whether use qself or kself emb.")
    parser.add_argument("--comb_loss", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None], default=True,
                        help="Whether use qself or kself emb.")
    parser.add_argument("--is_freeze_bert", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None],
                        default=True, help="Whether freeze bert.")
    parser.add_argument("--head_nums", type=int, default=4, help="head of nums for att.")
    parser.add_argument("--edgebert_hidden_size", type=int, default=1024, help="edgebert hidden size")

    parser.add_argument("--debug", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None], default=False,
                        help="debug mode")
    parser.add_argument("--evaluate_every_epoch", action="store_true", help="debug mode")

    parser.add_argument("--load_pretrained_bert_checkpoint", action="store_true",
                        help="load the previous baseline checkpoint")

    parser.add_argument("--inference_neighbor_bert", type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None],
                        default=False,
                        help="if donnot freeze bert, only grad the qk edge pair, inference mode for neighbors")

    parser.add_argument("--processed_cached_features_file_path", type=str, default=None,
                        help="default cached file path")

    parser.add_argument("--update_freezed_bert_parameter", action="store_true", help="update_freezed_bert_parameter")
    parser.add_argument("--update_freezed_bert_parameter_steps", type=int, default=500,
                        help="update_freezed_bert_parameter")

    # args for token-net part
    parser.add_argument("--gnn_token_embedding_path", type=str, default="",
                        help="pretrained gnn_token_embedding_path same shape as words embeddings in bert")
    parser.add_argument("--is_freeze_gnn_token_embedding",
                        type=lambda s: s.lower() in ['true', 't', 'yes', '1', '', None], default=True,
                        help="Whether to freeze the the gnn token embedding part")

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

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    ####################################################################
    # TODO update this part modify the model
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    # config = config_class(
    #     edgebert_hidden_size=args.edgebert_hidden_size,
    #     use_kkqq=args.use_kkqq,
    #     use_qk=args.use_qk,
    #     use_node=args.use_node,
    #     gat_acts=args.gat_acts,
    #     aggregator=args.aggregator,
    #     head_nums=args.head_nums,
    #     comb_loss=args.comb_loss,
    #     use_residual=args.use_residual,
    #     num_labels=num_labels,
    #     bert_model_path_or_name=args.model_name_or_path,
    #     is_freeze_bert=args.is_freeze_bert,
    #     inference_neighbor_bert=args.inference_neighbor_bert,
    #     gnn_token_embedding_path=args.gnn_token_embedding_path,
    #     is_freeze_gnn_token_embedding=args.is_freeze_gnn_token_embedding,
    # )
    #
    def get_current_model_config_from_args(current_args):
        tmp_config = config_class(
            edgebert_hidden_size=current_args.edgebert_hidden_size,
            use_kkqq=current_args.use_kkqq,
            use_qk=current_args.use_qk,
            use_node=current_args.use_node,
            gat_acts=current_args.gat_acts,
            aggregator=current_args.aggregator,
            head_nums=current_args.head_nums,
            comb_loss=current_args.comb_loss,
            use_residual=current_args.use_residual,
            num_labels=num_labels,
            bert_model_path_or_name=current_args.model_name_or_path,
            is_freeze_bert=current_args.is_freeze_bert,
            inference_neighbor_bert=current_args.inference_neighbor_bert,
            gnn_token_embedding_path=current_args.gnn_token_embedding_path,
            is_freeze_gnn_token_embedding=current_args.is_freeze_gnn_token_embedding,

        )
        return tmp_config

    config = get_current_model_config_from_args(current_args=args)

    # use default BERT tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class(config)

    ##################################################################

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    # logger.info(model)
    logger.info(config)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, tb_writer=tb_writer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.output_dir)
        torch.save(model_to_save.state_dict(), args.output_dir + '/model.bin')  # save the model state_dict file
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        with open(os.path.join(args.output_dir, "training_args.txt"), 'w', encoding='utf-8') as f_args:
            f_args.write(str(args.__dict__))
        # Load a trained model and vocabulary that you have fine-tuned
        # model = model_class.from_pretrained(args.output_dir)

        model = model_class(config)
        model.load_state_dict(torch.load(args.output_dir + '/model.bin'))
        # because we use the huggerface tokenizer
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)

        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = []
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + 'model.bin', recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        # checkpoints.append(args.output_dir)
        for checkpoint in checkpoints:
            print(checkpoint)
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            # prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            # model = model_class.from_pretrained(checkpoint)
            #
            model = model_class(config)
            print("load checkpoint : {}".format(checkpoint + '/model.bin'))
            model.load_state_dict(torch.load(checkpoint + '/model.bin'))

            model.to(args.device)
            prefix = ""
            if global_step != "" and global_step.isdigit():
                prefix = "checkpoint-{}".format(global_step)

            print(prefix)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    # Inference
    if args.do_inference and args.local_rank in [-1, 0]:

        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        infer_checkpoints = []
        if args.inference_all_checkpoints:
            infer_checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + 'model.bin', recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        else:
            infer_checkpoints.append(args.output_dir)
        logger.info("Evaluate the following checkpoints: %s", infer_checkpoints)
        # checkpoints.append(args.output_dir)
        for checkpoint in infer_checkpoints:
            print(checkpoint)
            global_step = checkpoint.split("-")[-1] if len(infer_checkpoints) > 1 else ""
            # get current args
            if os.path.exists(checkpoint + "/" + "training_args.bin"):
                print("load previous args : {}".format(checkpoint + "/" + "training_args.bin"))
                inference_args = torch.load(checkpoint + "/" + "training_args.bin")
            else:
                print("risky, use current args, may fail")
                inference_args = args
            # check the same task_name (for same data processor)
            assert inference_args.task_name == args.task_name
            inference_config = get_current_model_config_from_args(inference_args)
            model = model_class(inference_config)
            print("load checkpoint : {}".format(checkpoint + '/model.bin'))
            model.load_state_dict(torch.load(checkpoint + '/model.bin'))
            model.to(args.device)  # NOTICE : Not inference_args
            prefix = ""
            if global_step != "" and global_step.isdigit():
                prefix = "checkpoint-{}".format(global_step)

            print(prefix)
            inference(args, model, tokenizer, output_folder=checkpoint)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return results


if __name__ == "__main__":
    main()