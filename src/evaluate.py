import torch
import torch.nn as nn
import numpy as np
import json
from tqdm import tqdm
from utils import is_one_stage, is_two_stage
from sklearn.metrics import classification_report
import config


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
        doc_id, word_id, tokens, attention_mask, token_type_ids, l_mask, r_mask, labels, pn_labels = data
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
    id2hier = {1: 5, 2: 1, 3: 2, 4: 5, 5: 3, 6: 5, 7: 4, 8: 1, 9: 3, 10: 5, 11: 3, 12: 3, 13: 3, 
            14: 5, 15: 3, 16: 3, 17: 4, 18: 5, 19: 5, 20: 5, 21: 3, 22: 5, 23: 5, 24: 4, 25: 5, 26: 5, 27: 3, 
            28: 5, 29: 3, 30: 3, 31: 2, 32: 3, 33: 5, 34: 3, 35: 1, 36: 5, 37: 3, 38: 1, 39: 5, 40: 1, 41: 3, 
            42: 1, 43: 5, 44: 5, 45: 5, 46: 5, 47: 3, 48: 3, 49: 3, 50: 5, 51: 5, 52: 1, 53: 1, 54: 3, 55: 1, 
            56: 5, 57: 5, 58: 4, 59: 5, 60: 5, 61: 2, 62: 5, 63: 3, 64: 5, 65: 3, 66: 5, 67: 5, 68: 4, 69: 5, 
            70: 5, 71: 5, 72: 5, 73: 1, 74: 5, 75: 1, 76: 5, 77: 5, 78: 4, 79: 5, 80: 5, 81: 5, 82: 5, 83: 5, 
            84: 2, 85: 5, 86: 1, 87: 5, 88: 3, 89: 5, 90: 3, 91: 5, 92: 4, 93: 4, 94: 5, 95: 1, 96: 3, 97: 5, 
            98: 1, 99: 3, 100: 5, 101: 5, 102: 3, 103: 5, 104: 5, 105: 5, 106: 5, 107: 2, 108: 5, 109: 5, 110: 2, 
            111: 4, 112: 5, 113: 5, 114: 5, 115: 1, 116: 4, 117: 3, 118: 5, 119: 3, 120: 1, 121: 5, 122: 1, 123: 1, 
            124: 4, 125: 5, 126: 3, 127: 2, 128: 5, 129: 3, 130: 4, 131: 3, 132: 5, 133: 5, 134: 3, 135: 5, 136: 5, 
            137: 3, 138: 5, 139: 3, 140: 2, 141: 5, 142: 5, 143: 4, 144: 1, 145: 1, 146: 4, 147: 1, 148: 1, 149: 5, 
            150: 5, 151: 5, 152: 1, 153: 5, 154: 4, 155: 5, 156: 5, 157: 2, 158: 5, 159: 5, 160: 1, 161: 2, 162: 1, 
            163: 3, 164: 5, 165: 3, 166: 5, 167: 2, 168: 1, 0: 0}
    id_masks = []
    for i in range(6):
        id_mask = [0] * config.maven_class_numbers
        for j in range(config.maven_class_numbers):
            if id2hier[j] == i:
                id_mask[j] = 1
        id_masks.append(np.array(id_mask))
    model.eval()

    print('Testing:')
    pred_list = []
    doc_ids   = []
    word_ids  = []
    for i, data in tqdm(enumerate(loader)):
        doc_id, word_id, tokens, attention_mask, token_type_ids, l_mask, r_mask, _, _ = data
        doc_ids += doc_id
        word_ids += word_id
        if args.cuda:
            tokens = tokens.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()
            l_mask = l_mask.cuda()
            r_mask = r_mask.cuda()
        
        if is_two_stage(args.model):
            logits_tf, logits_cl = model(tokens, attention_mask, token_type_ids, l_mask, r_mask)
            tf_pred = torch.argmax(logits_tf, dim=1).tolist()
            # cl_pred = torch.argmax(logits_cl, dim=1).tolist()
            cl_pred = [0]*logits_cl.shape[0]
            for i in range(len(tf_pred)):
                if tf_pred[i] != 0:
                    cl_pred[i] = torch.argmax(logits_cl[i]*id_masks[tf_pred[i]])
            pred_list += cl_pred
        # elif is_one_stage(args.model):
        #     logits = model(tokens)
        #     pred = torch.argmax(logits, dim=1).tolist()
        #     pred_list += pred
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
