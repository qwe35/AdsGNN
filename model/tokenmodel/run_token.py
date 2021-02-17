import torch
import torch.optim as optim
import utils
import os
from models.topo_modeling import TopoGramForNeighborPredict
from models.bi_topo_modeling import BiTopoGramForNeighborPredict
from models.configuration_tnlrv3 import TuringNLRv3Config
import copy

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import time
import logging

from data_handler_4_graph_only_title import DatasetForMatching, DataCollatorForMatching,SingleProcessDataLoaderForMatching,MultiProcessDataLoaderForMatching
from transformers import BertTokenizerFast

from run_retrive import compute_acc, compute_retrive_acc, setup, cleanup, warmup_linear

def load_bert(args):
    config = TuringNLRv3Config.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True)
    config.neighbor_type = args.neighbor_type
    config.mapping_graph = 1 if args.mapping_graph else 0
    config.graph_transform = 1 if args.return_last_station_emb else 0
    model = TopoGramForNeighborPredict.from_pretrained(args.model_name_or_path,
                                                       from_tf=bool('.ckpt' in args.model_name_or_path),
                                                       config=config)
    # model = BiTopoGramForNeighborPredict.from_pretrained(args.model_name_or_path,
    #                                                    from_tf=bool('.ckpt' in args.model_name_or_path),
    #                                                    config=config)
    return model

def train(local_rank, args, global_prefetch_step, end):

    utils.setuplogging()
    os.environ["RANK"] = str(local_rank)
    setup(local_rank, args.world_size)
    device = torch.device("cuda", local_rank)
    if args.fp16:
        from torch.cuda.amp import autocast
        scaler = torch.cuda.amp.GradScaler()

    model = load_bert(args)
    logging.info('loading model: {}'.format(args.model_type))
    model = model.to(device)
    if args.world_size > 1:
        ddp_model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    else:
        ddp_model = model

    if args.warmup_lr:
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr*warmup_linear(args,0)}])
    else:
        optimizer = optim.Adam([{'params': ddp_model.parameters(), 'lr': args.pretrain_lr}])

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorForMatching(tokenizer=tokenizer,mlm=args.mlm_loss,neighbor_num=args.neighbor_num,neighbor_mask=args.neighbor_mask,block_size=args.block_size)
    loss = 0.0
    global_step = 0
    best_acc=0.0
    best_model=copy.deepcopy(model)
    for ep in range(args.epochs):
        start_time = time.time()
        ddp_model.train()
        dataset = DatasetForMatching(tokenizer=tokenizer,file_path=args.train_data_path,
                                                   neighbor_num=args.neighbor_num)
        dataloader = MultiProcessDataLoaderForMatching(dataset,
                                                         batch_size=args.train_batch_size,
                                                         collate_fn=data_collator,
                                                         local_rank=local_rank,
                                                         world_size=args.world_size,
                                                         prefetch_step=global_prefetch_step,
                                                         end=end)
        for step, batch in enumerate(dataloader):

            if args.enable_gpu:
                for k,v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_id_query = batch['input_id_query']
            attention_masks_query = batch['attention_masks_query']
            masked_lm_labels_query = batch['masked_lm_labels_query']
            mask_query = batch['mask_query']
            input_id_key = batch['input_id_key']
            attention_masks_key = batch['attention_masks_key']
            masked_lm_labels_key = batch['masked_lm_labels_key']
            mask_key = batch['mask_key']
            if args.fp16:
                with autocast():
                    batch_loss = ddp_model(
                        input_id_query,
                        attention_masks_query,
                        masked_lm_labels_query,
                        mask_query,
                        input_id_key,
                        attention_masks_key,
                        masked_lm_labels_key,
                        mask_key,
                        neighbor_num=args.neighbor_num,
                        mask_self_in_graph=args.self_mask,
                        mlm_loss=args.mlm_loss,
                        return_last_station_emb=args.return_last_station_emb)
            else:
                batch_loss = ddp_model(
                    input_id_query,
                    attention_masks_query,
                    masked_lm_labels_query,
                    mask_query,
                    input_id_key,
                    attention_masks_key,
                    masked_lm_labels_key,
                    mask_key,
                    neighbor_num=args.neighbor_num,
                    mask_self_in_graph=args.self_mask,
                    mlm_loss=args.mlm_loss,
                    return_last_station_emb=args.return_last_station_emb)
            loss += batch_loss.item()
            optimizer.zero_grad()
            if args.fp16:
                scaler.scale(batch_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                batch_loss.backward()
                optimizer.step()

            global_step+=1
            if args.warmup_lr:
                optimizer.param_groups[0]['lr'] = args.pretrain_lr*warmup_linear(args,global_step)

            if local_rank == 0 and global_step % args.log_steps == 0:
                logging.info(
                    '[{}] cost_time:{} step:{}, lr:{}, train_loss: {:.5f}'.format(
                        local_rank, time.time()-start_time, global_step, optimizer.param_groups[0]['lr'],loss / args.log_steps))
                loss=0.0

            # save model minibatch
            if local_rank == 0 and global_step % args.save_steps == 0:
                ckpt_path = os.path.join(args.model_dir, f'{args.savename}-epoch-{ep}-{global_step}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")

            dist.barrier()
        logging.info("train time:{}".format(time.time() - start_time))

        # save model last of epoch
        if local_rank == 0:
            ckpt_path = os.path.join(args.model_dir, '{}-epoch-{}.pt'.format(args.savename,ep+1))
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

            logging.info("Star testing for epoch-{}".format(ep+1))
            acc=test_single_process(model,args,"valid")
            logging.info("test time:{}".format(time.time() - start_time))
            if acc>best_acc:
                best_model=copy.deepcopy(model)
                best_acc=acc
        dist.barrier()

    if local_rank==0:
        start_time=time.time()
        ckpt_path = os.path.join(args.model_dir, '{}-best.pt'.format(args.savename))
        torch.save(
            {
                'model_state_dict': best_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt_path)
        logging.info(f"Model saved to {ckpt_path}")
        logging.info("Star testing for best")
        acc = test_single_process(best_model, args,"test")
        logging.info("test time:{}".format(time.time() - start_time))

    cleanup()


def test_single_process(model,args,mode):
    assert mode in {"valid","test"}
    model.eval()
    with torch.no_grad():

        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        data_collator = DataCollatorForMatching(tokenizer=tokenizer, mlm=args.mlm_loss, neighbor_num=args.neighbor_num,
                                                neighbor_mask=args.neighbor_mask, block_size=args.block_size)
        if mode=="valid":
            dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.valid_data_path,
                                                   neighbor_num=args.neighbor_num)
            dataloader = SingleProcessDataLoaderForMatching(dataset,batch_size=args.valid_batch_size,collate_fn=data_collator)
        elif mode=="test":
            dataset = DatasetForMatching(tokenizer=tokenizer, file_path=args.test_data_path,
                                         neighbor_num=args.neighbor_num)
            dataloader = SingleProcessDataLoaderForMatching(dataset, batch_size=args.test_batch_size,
                                                            collate_fn=data_collator)

        mlm_acc = [0, 0]
        retrive_acc = [0, 0]
        for step, batch in enumerate(dataloader):
            if args.enable_gpu:
                for k, v in batch.items():
                    if v is not None:
                        batch[k] = v.cuda(non_blocking=True)

            input_ids_query = batch['input_id_query']
            attention_masks_query = batch['attention_masks_query']
            masked_lm_labels_query = batch['masked_lm_labels_query']
            mask_query = batch['mask_query']
            input_ids_key = batch['input_id_key']
            attention_masks_key = batch['attention_masks_key']
            masked_lm_labels_key = batch['masked_lm_labels_key']
            mask_key = batch['mask_key']

            all_nodes_num = mask_query.shape[0]
            batch_size = all_nodes_num // (args.neighbor_num + 1)
            neighbor_mask_query = mask_query.view(batch_size, (args.neighbor_num + 1))
            neighbor_mask_key = mask_key.view(batch_size, (args.neighbor_num + 1))

            hidden_states_query = model.bert(input_ids_query, attention_masks_query,
                                                 neighbor_mask=neighbor_mask_query,
                                                 mask_self_in_graph=args.self_mask,
                                                 return_last_station_emb=args.return_last_station_emb
                                                 )
            hidden_states_key = model.bert(input_ids_key, attention_masks_key,
                                               neighbor_mask=neighbor_mask_key,
                                               mask_self_in_graph=args.self_mask,
                                               return_last_station_emb=args.return_last_station_emb
                                               )
            last_hidden_states_query = hidden_states_query[0]
            last_hidden_states_key = hidden_states_key[0]

            if args.neighbor_type != 0:
                # delete the station_placeholder hidden_state:(N,1+L,D)->(N,L,D)
                last_hidden_states_query = last_hidden_states_query[:, 1:]
                last_hidden_states_key = last_hidden_states_key[:, 1:]

            # hidden_state:(N,L,D)->(B,L,D)
            query = last_hidden_states_query[::(args.neighbor_num + 1)]
            key = last_hidden_states_key[::(args.neighbor_num + 1)]
            if masked_lm_labels_query is not None:
                mlm_scores = model.cls.predictions(query, model.bert.embeddings.word_embeddings.weight)  # N L V
                hit_num, all_num = compute_acc(mlm_scores, masked_lm_labels_query)
                mlm_acc[0] += hit_num.data
                mlm_acc[1] += all_num.data

            if masked_lm_labels_key is not None:
                mlm_scores = model.cls.predictions(key, model.bert.embeddings.word_embeddings.weight)  # N L V
                hit_num, all_num = compute_acc(mlm_scores, masked_lm_labels_key)
                mlm_acc[0] += hit_num.data
                mlm_acc[1] += all_num.data

            mask_query = mask_query[::(args.neighbor_num + 1)]
            mask_key = mask_key[::(args.neighbor_num + 1)]

            if args.return_last_station_emb:
                last_neighbor_hidden_states_query = hidden_states_query[-1]
                last_neighbor_hidden_states_key = hidden_states_key[-1]
                query = torch.cat([query[:, 0], last_neighbor_hidden_states_query], dim=-1)
                query = model.graph_transform(query)
                key = torch.cat([key[:, 0], last_neighbor_hidden_states_key], dim=-1)
                key = model.graph_transform(key)
            else:
                query = query[:, 0]
                key = key[:, 0]

            # hit_num, all_num = compute_retrive_acc(query, key, mask_q=mask_query, mask_k=mask_key, Q=tokenizer.batch_decode(
            #     input_ids_query[::(args.neighbor_num + 1)].cpu().numpy().tolist(), skip_special_tokens=True),
            #                                        K=tokenizer.batch_decode(
            #                                            input_ids_key[::(args.neighbor_num + 1)].cpu().numpy().tolist(),
            #                                            skip_special_tokens=True))
            hit_num, all_num = compute_retrive_acc(query, key, mask_q=mask_query, mask_k=mask_key)
            retrive_acc[0] += hit_num.data
            retrive_acc[1] += all_num.data

        if args.mlm_loss:
            logging.info('Final-- mlm_acc:{}, qk_acc:{}'.format((mlm_acc[0] / mlm_acc[1]).data,(retrive_acc[0] / retrive_acc[1]).data))
        else:
            logging.info('Final-- qk_acc:{}'.format((retrive_acc[0] / retrive_acc[1]).data))
        return (retrive_acc[0] / retrive_acc[1]).data



# def test(local_rank, args, global_prefetch_step, end):
#
#     utils.setuplogging()
#     os.environ["RANK"] = str(local_rank)
#     setup(local_rank, args.world_size)
#
#     device = torch.device("cuda", local_rank)
#
#     model = load_bert(args)
#     logging.info('loading model: {}'.format(args.model_type))
#     model = model.to(device)
#
#     checkpoint = torch.load(args.load_ckpt_name)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     logging.info('load ckpt:{}'.format(args.load_ckpt_name))
#
#     if args.world_size > 1:
#         ddp_model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
#     else:
#         ddp_model = model
#
#     ddp_model.eval()
#     torch.set_grad_enabled(False)
#
#     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
#     dataset = DatasetForSentencePairPrediction(tokenizer=tokenizer, file_path=args.train_data_path,
#                                                neighbor_num=args.neighbor_num)
#     data_collator = DataCollatorForSentencePairPrediction(tokenizer=tokenizer, block_size=args.block_size,mlm=False)
#     dataloader = DataLoaderForSentencePairPrediction(dataset,
#                                                      batch_size=args.batch_size,
#                                                      collate_fn=data_collator,
#                                                      local_rank=local_rank,
#                                                      world_size=args.world_size,
#                                                      prefetch_step=global_prefetch_step,
#                                                      end=end)
#
#     mlm_acc = [0,0]
#     retrive_acc = [0,0]
#     for step, batch in enumerate(dataloader):
#         if args.enable_gpu:
#             for k, v in batch.items():
#                 if v is not None:
#                     batch[k] = v.cuda(non_blocking=True)
#
#         input_ids_query = batch['input_id_query']
#         attention_masks_query = batch['attention_masks_query']
#         masked_lm_labels_query = batch['masked_lm_labels_query']
#         mask_query = batch['mask_query']
#         input_ids_key = batch['input_id_key']
#         attention_masks_key = batch['attention_masks_key']
#         masked_lm_labels_key = batch['masked_lm_labels_key']
#         mask_key = batch['mask_key']
#
#         all_nodes_num = mask_query.shape[0]
#         batch_size = all_nodes_num // (args.neighbor_num + 1)
#         neighbor_mask_query = mask_query.view(batch_size, (args.neighbor_num + 1))
#         neighbor_mask_key = mask_key.view(batch_size, (args.neighbor_num + 1))
#
#         hidden_states_query = ddp_model.bert(input_ids_query, attention_masks_query,
#                                         neighbor_mask=neighbor_mask_query,
#                                         mask_self_in_graph=args.self_mask,
#                                         return_last_station_emb=args.return_last_station_emb
#                                         )
#         hidden_states_key = ddp_model.bert(input_ids_key, attention_masks_key,
#                                       neighbor_mask=neighbor_mask_key,
#                                       mask_self_in_graph=args.self_mask,
#                                       return_last_station_emb=args.return_last_station_emb
#                                       )
#         last_hidden_states_query = hidden_states_query[0]
#         last_hidden_states_key = hidden_states_key[0]
#
#         if args.neighbor_type != 0 :
#             # delete the station_placeholder hidden_state:(N,1+L,D)->(N,L,D)
#             last_hidden_states_query = last_hidden_states_query[:, 1:]
#             last_hidden_states_key = last_hidden_states_key[:, 1:]
#
#         # hidden_state:(N,L,D)->(B,L,D)
#         query = last_hidden_states_query[::(args.neighbor_num + 1)]
#         key = last_hidden_states_key[::(args.neighbor_num + 1)]
#
#         mlm_scores = ddp_model.cls.predictions(query, ddp_model.bert.embeddings.word_embeddings.weight) #N L V
#         hit_num, all_num = compute_acc(mlm_scores,masked_lm_labels_query)
#         mlm_acc[0] += hit_num.data
#         mlm_acc[1] += all_num.data
#
#         mlm_scores = ddp_model.cls.predictions(key, ddp_model.bert.embeddings.word_embeddings.weight) #N L V
#         hit_num, all_num = compute_acc(mlm_scores,masked_lm_labels_key)
#         mlm_acc[0] += hit_num.data
#         mlm_acc[1] += all_num.data
#
#         mask_query = mask_query[::(args.neighbor_num + 1)]
#         mask_key = mask_key[::(args.neighbor_num + 1)]
#
#         if args.return_last_station_emb:
#             last_neighbor_hidden_states_query = hidden_states_query[-1]
#             last_neighbor_hidden_states_key = hidden_states_key[-1]
#             query = torch.cat([query[:, 0], last_neighbor_hidden_states_query], dim=-1)
#             query = ddp_model.graph_transform(query)
#             key = torch.cat([key[:, 0], last_neighbor_hidden_states_key], dim=-1)
#             key = ddp_model.graph_transform(key)
#
#         else:
#             query = query[:,0]
#             key = key[:,0]
#
#         hit_num, all_num = compute_retrive_acc(query, key, mask_q=mask_query, mask_k=mask_key,Q=tokenizer.batch_decode(input_ids_query[::(args.neighbor_num + 1)].cpu().numpy().tolist(),skip_special_tokens=True),
#                                                K=tokenizer.batch_decode(input_ids_key[::(args.neighbor_num + 1)].cpu().numpy().tolist(),skip_special_tokens=True))
#         # hit_num, all_num = compute_retrive_acc(query, key, mask_q=mask_query, mask_k=mask_key)
#         retrive_acc[0] += hit_num.data
#         retrive_acc[1] += all_num.data
#
#         if step%args.log_steps == 0:
#             logging.info('[{}] step:{}, mlm_acc:{}, qk_acc:{}'.format(local_rank,step,
#                 (mlm_acc[0]/mlm_acc[1]).data, (retrive_acc[0]/retrive_acc[1]).data
#             ))
#             # logging.info('[{}] step:{}, qk_acc:{}'.format(local_rank, step, (retrive_acc[0] / retrive_acc[1]).data))
#
#     logging.info('Final-- [{}] mlm_acc:{}, qk_acc:{}'.format(local_rank,
#         (mlm_acc[0] / mlm_acc[1]).data, (retrive_acc[0] / retrive_acc[1]).data
#     ))
#
#     cleanup()


