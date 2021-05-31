import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
from utils import is_one_stage, is_two_stage
from sklearn.metrics import classification_report


@torch.no_grad()
def evaluate_one_stage_model(args, model, loader):
    pass


@torch.no_grad()
def evaluate_two_stage_model(args, model, loader):
    criterion_tf = nn.CrossEntropyLoss()
    criterion_cl = nn.CrossEntropyLoss(ignore_index=0)  # ignore 0 type
    model.eval()

    print('Evaluating:')
    loss_tf_list = []
    loss_cl_list = []
    pred_list = []
    gt_labels = []
    doc_ids   = []
    word_ids  = []
    for i, data in tqdm(enumerate(loader)):
        doc_id, word_id, tokens, labels, pn_labels = data
        if args.cuda:
            tokens = tokens.cuda()
            labels = labels.cuda()
            pn_labels = pn_labels.cuda()
        logits_tf, logits_cl = model(tokens)
        loss_tf = criterion_tf(logits_tf, pn_labels)
        loss_cl = criterion_cl(logits_cl, labels)
        loss_tf_list.append(loss_tf.item())
        loss_cl_list.append(loss_cl.item())
        
        doc_ids += doc_id
        word_ids += word_id
        gt_labels += labels.tolist()
        tf_pred = torch.argmax(logits_tf, dim=1).tolist()
        cl_pred = torch.argmax(logits_cl, dim=1).tolist()
        for i in range(len(tf_pred)):
            if tf_pred[i] == 0:
                cl_pred[i] = 0
        pred_list += cl_pred
    
    print('Average loss_tf: {}'.format(np.mean(loss_tf_list)))
    print('Average loss_cl: {}'.format(np.mean(loss_cl_list)))
    print(classification_report(gt_labels, pred_list, digits=4))


@torch.no_grad()
def test(args, model, loader):
    model.eval()

    print('Testing:')
    pred_list = []
    doc_ids   = []
    word_ids  = []
    for i, data in tqdm(enumerate(loader)):
        doc_id, word_id, tokens, _, _ = data
        doc_ids += doc_id
        word_ids += word_id
        if args.cuda:
            tokens = tokens.cuda()
        
        if is_two_stage(args.model):
            logits_tf, logits_cl = model(tokens)
            tf_pred = torch.argmax(logits_tf, dim=1).tolist()
            cl_pred = torch.argmax(logits_cl, dim=1).tolist()
            for i in range(len(tf_pred)):
                if tf_pred[i] == 0:
                    cl_pred[i] = 0
            pred_list += cl_pred
        elif is_one_stage(args.model):
            logits = model(tokens)
            pred = torch.argmax(logits, dim=1).tolist()
            pred_list += pred
        else:
            return NotImplementedError()
    return doc_ids, word_ids, pred_list


def get_submission(args, model, loader):
    doc_ids, word_ids, pred_list = test(args, model, loader)
    pred_obj = {}
    for doc_id, word_id, pred in zip(doc_ids, word_ids, pred_list):
        if doc_id not in pred_obj:
            pred_obj[doc_id] = {
                'id': doc_id,
                'predictions': []
            }
        pred_obj[doc_id]['predictions'].append({
            'id': word_id,
            'type_id': pred
        })
    jsonl_str = ''
    for k, v in pred_obj.items():
        jsonl_str += json.dumps(v) + '\n'
    return jsonl_str
