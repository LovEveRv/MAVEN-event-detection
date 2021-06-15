import os
import torch
import torch.nn as nn
import config
import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from utils import save_checkpoint
from evaluate import evaluate_one_stage_model, evaluate_two_stage_model


def train_one_stage_model(args, model, optimizer, train_loader, val_loader, epochs=10):
    pass


def train_two_stage_model(args, model, optimizer, train_loader, val_loader, epochs=10):
    criterion_tf = nn.CrossEntropyLoss()
    criterion_cl = nn.CrossEntropyLoss(ignore_index=0)  # ignore 0 type
    model.train()

    for epoch in range(1, epochs + 1):
        print('Training epoch {}/{}:'.format(epoch, epochs))
        epoch_loss_tf_list = []
        epoch_loss_cl_list = []
        tmp_loss_tf_list = []
        tmp_loss_cl_list = []
        for i, data in tqdm(enumerate(train_loader)):
            _, _, tokens, attention_mask, token_type_ids, l_mask, r_mask, labels, pn_labels = data
            if args.cuda:
                tokens = tokens.cuda()
                attention_mask = attention_mask.cuda()
                token_type_ids = token_type_ids.cuda()
                l_mask = l_mask.cuda()
                r_mask = r_mask.cuda()
                labels = labels.cuda()
                pn_labels = pn_labels.cuda()
            logits_tf, logits_cl = model(tokens, attention_mask, token_type_ids, l_mask, r_mask)
            loss_tf = criterion_tf(logits_tf, pn_labels)
            loss_cl = criterion_cl(logits_cl, labels)
            epoch_loss_tf_list.append(loss_tf.item())
            epoch_loss_cl_list.append(loss_cl.item())
            tmp_loss_tf_list.append(loss_tf.item())
            tmp_loss_cl_list.append(loss_cl.item())
            tr_loss = (loss_tf + loss_cl) / args.grad_acum_step
            tr_loss.backward()
            if (i + 1) % args.grad_acum_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            if args.logging > 0 and (i + 1) % args.logging == 0:
                print('loss_tf = {}'.format(np.mean(tmp_loss_tf_list)))
                print('loss_cl = {}'.format(np.mean(tmp_loss_cl_list)))
                tmp_loss_tf_list = []
                tmp_loss_cl_list = []
        
        print('Epoch average loss_tf: {}'.format(np.mean(epoch_loss_tf_list)))
        print('Epoch average loss_cl: {}'.format(np.mean(epoch_loss_cl_list)))
        evaluate_two_stage_model(args, model, val_loader)
        if args.local_rank == 0:
            # save checkpoint
            ckpt_path = os.path.join(config.checkpoint_dir, 'bert-two-stage-{}.pkl'.format(epoch))
            if args.distributed:
                save_checkpoint(ckpt_path, model.module, optimizer)
            else:
                save_checkpoint(ckpt_path, model, optimizer)
        if args.distributed:
            torch.distributed.barrier()
