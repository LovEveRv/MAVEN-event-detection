import os
import torch
import torch.utils.data as data
import config


class MavenSet(data.Dataset):
    """
    MAVEN Dataset
    """

    def __init__(self, root, split='train'):
        super().__init__()
        self.split = split
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

    def _gen_samples(self, data):
        samples = []
        for doc in data:
            doc_id = doc['id']
            content = doc['content']
            if self.split != 'test':
                for event in doc['events']:
                    type_id = event['type_id']
                    for word in event['mention']:
                        tokens = content[word['sent_id']]['tokens']
                        offset = word['offset']
                        tokens = self._add_special_tokens(tokens, offset)
                        samples.append({
                            'doc_id': doc_id,
                            'word_id': word['id'],
                            'tokens': tokens,
                            'type': type_id
                        })
                for word in doc['negative_triggers']:
                    type_id = 0  # marks negative
                    tokens = content[word['sent_id']]['tokens']
                    offset = word['offset']
                    tokens = self._add_special_tokens(tokens, offset)
                    samples.append({
                        'doc_id': doc_id,
                        'word_id': word['id'],
                        'tokens': tokens,
                        'type': type_id
                    })
            else:
                for word in doc['candidates']:
                    type_id = None  # need to predict
                    tokens = content[word['sent_id']]['tokens']
                    offset = word['offset']
                    tokens = self._add_special_tokens(tokens, offset)
                    samples.append({
                        'doc_id': doc_id,
                        'word_id': word['id'],
                        'tokens': tokens,
                        'type': type_id
                    })
        return samples

    def _add_special_tokens(self, tokens, offset):
        head = offset[0]
        tail = offset[1]
        new_tokens = tokens[:head] + [config.trigger_start_token] + tokens[head:tail] + [config.trigger_end_token] + tokens[tail:]
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
        return sample['doc_id'], sample['word_id'], sample['tokens'], sample['type']


if __name__ == '__main__':
    """
    For test only
    """
    dataset = MavenSet('../MAVEN', 'test')
    # dataset.stat()
    print(len(dataset))
    print(dataset[0])
    print(dataset[1])
    # dataset = MavenSet('../MAVEN', 'val')
    # dataset.stat()
    # print(len(dataset))
    # dataset = MavenSet('../MAVEN', 'train')
    # dataset.stat()
    # print(len(dataset))
