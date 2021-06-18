import os
import torch
import torch.utils.data as data
import config


class MavenSet(data.Dataset):
    """
    MAVEN Dataset
    """

    def __init__(self, root, tokenizer, split='train'):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        if split == 'train':
            file_path = os.path.join(root, 'train.jsonl')
        elif split == 'val':
            file_path = os.path.join(root, 'valid.jsonl')
        elif split == 'test':
            file_path = os.path.join(root, 'test.jsonl')
        else:
            raise NotImplementedError('"split" must be "train" or "val" or "test".')
        
        self.rawdata = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) > 0:
                    self.rawdata.append(eval(line))
        
        self.data = self._gen_samples(self.rawdata)
        self.id2hier = {1: 5, 2: 1, 3: 2, 4: 5, 5: 3, 6: 5, 7: 4, 8: 1, 9: 3, 10: 5, 11: 3, 12: 3, 13: 3, 
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

    def _gen_samples(self, rawdata):
        samples = []
        for doc in rawdata:
            doc_id = doc['id']
            content = doc['content']
            if self.split != 'test':
                for event in doc['events']:
                    type_id = event['type_id']
                    for word in event['mention']:
                        tokens = content[word['sent_id']]['tokens']
                        offset = word['offset']
                        # tokens = self._add_special_tokens(tokens, offset)
                        samples.append({
                            'doc_id': doc_id,
                            'word_id': word['id'],
                            'tokens': tokens,
                            'offset': offset,
                            'type': type_id
                        })
                for word in doc['negative_triggers']:
                    type_id = 0  # marks negative
                    tokens = content[word['sent_id']]['tokens']
                    offset = word['offset']
                    # tokens = self._add_special_tokens(tokens, offset)
                    samples.append({
                        'doc_id': doc_id,
                        'word_id': word['id'],
                        'tokens': tokens,
                        'offset': offset,
                        'type': type_id
                    })
            else:
                for word in doc['candidates']:
                    type_id = -100  # need to predict
                    tokens = content[word['sent_id']]['tokens']
                    offset = word['offset']
                    # tokens = self._add_special_tokens(tokens, offset)
                    samples.append({
                        'doc_id': doc_id,
                        'word_id': word['id'],
                        'tokens': tokens,
                        'offset': offset,
                        'type': type_id
                    })
        return samples

    def _add_special_tokens(self, tokens, offset):  # not used anymore
        head = offset[0]
        tail = offset[1]
        # convert to lower case
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        new_tokens = [self.tokenizer.cls_token] + tokens[:head] + [config.trigger_start_token] + \
            tokens[head:tail] + [config.trigger_end_token] + tokens[tail:] + [self.tokenizer.sep_token]
        return new_tokens

    def stat(self):
        # Find max sentence length and max document length
        max_sentence_len = 0
        max_content_len = 0
        for sample in self.rawdata:
            content = sample['content']
            content_len = 0
            for sent in content:
                sent_len = len(sent['tokens'])
                max_sentence_len = max(max_sentence_len, sent_len)
                content_len += sent_len
            max_content_len = max(max_content_len, content_len)
        print('MAX sentence length: {} tokens'.format(max_sentence_len))
        print('MAX content  length: {} tokens'.format(max_content_len))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        pn_label = self.id2hier[sample['type']] # 1 for positive, 0 for negative
        head = sample['offset'][0]
        tail = sample['offset'][1]
        max_length = config.sentence_max_length
        tokens = sample['tokens']
        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()
        l_text = self.tokenizer.tokenize(' '.join(tokens[:head]))
        r_text = self.tokenizer.tokenize(' '.join(tokens[head:tail])) + [config.trigger_end_token] + self.tokenizer.tokenize(' '.join(tokens[tail:]))
        l_mask = [1.0 for _ in range(len(l_text) + 1)] + [0.0 for i in range(len(r_text) + 2)]
        r_mask = [0.0 for _ in range(len(l_text) + 1)] + [1.0 for i in range(len(r_text) + 2)]
        if len(l_mask) > max_length:
            l_mask = l_mask[:max_length]
        if len(r_mask) > max_length:
            r_mask = r_mask[:max_length]
        inputs = self.tokenizer.encode_plus(
            l_text + [config.trigger_start_token] + r_text, add_special_tokens=True, truncation=True, max_length=max_length, return_token_type_ids=True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)

        # padding
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        l_mask = l_mask + ([0.0] * padding_length)
        r_mask = r_mask + ([0.0] * padding_length)
        
        ret = (
            sample['doc_id'],
            sample['word_id'],
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(token_type_ids, dtype=torch.long),
            torch.tensor(l_mask),
            torch.tensor(r_mask),
            sample['type'],
            pn_label,
        )
        return ret


def MavenLoader(root, tokenizer, split, batch_size, num_workers=0):
    dataset = MavenSet(root, tokenizer, split)
    shuffle = split == 'train'
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers
    )


def DistributedMavenLoader(root, tokenizer, split, batch_size, num_workers=0):
    dataset = MavenSet(root, tokenizer, split)
    shuffle = split == 'train'
    distributed_sampler = data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        sampler=distributed_sampler
    )


if __name__ == '__main__':
    """
    For test only
    """  
    # dataset = MavenSet('../MAVEN', 'test')
    # print(len(dataset))
    # print(dataset[0])
    # print(dataset[1])
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    loader = MavenLoader('../MAVEN', tokenizer, 'test', 4)
    for i, data in enumerate(loader):
        doc_ids, word_ids, tokens, labels, pn_labels = data
        print(tokens)
        break
