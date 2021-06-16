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
        pn_label = 1 if sample['type'] > 0 else 0  # 1 for positive, 0 for negative
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
