"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
from collections import defaultdict

from tacred_utils import constant, helper, vocab


class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """

    def __init__(self, filename, batch_size, opt, vocab, evaluation=False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.batch_index = 0

        with open(filename) as infile:
            data = json.load(infile)
        data = self.preprocess(data, vocab, opt)
        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])
        self.labels = [id2label[d['relation']] for d in data]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

        data = self.batch_pad_data(data)

        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def pad_sample(self, sample, max_size):
        padded_sample = sample + [constant.PAD_ID] * (max_size - len(sample))
        return padded_sample

    def batch_pad_data(self, data):
        padded_data = []
        for batch in data:
            max_sentence_len = 0
            batch_features = defaultdict(list)
            for sample in batch:
                sentence_len = len(sample['tokens'])

                if max_sentence_len < sentence_len:
                    max_sentence_len = sentence_len

                for name, values in sample.items():
                    batch_features[name].append(values)

            for feature, values in batch_features.items():
                if feature == 'relation': continue
                for i, value in enumerate(values):
                    batch_features[feature][i] = self.pad_sample(value, max_sentence_len)
                batch_features[feature] = np.array(batch_features[feature])
            padded_data.append(batch_features)

        return padded_data

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        for d in data:
            tokens = d['token']
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se + 1] = ['SUBJ-' + d['subj_type']] * (se - ss + 1)
            tokens[os:oe + 1] = ['OBJ-' + d['obj_type']] * (oe - os + 1)
            tokens = map_to_ids(tokens, vocab.word2id)

            stanford_pos = d['stanford_pos']
            stanford_ner = d['stanford_ner']
            stanford_deprel = d['stanford_deprel']

            pos = map_to_ids(stanford_pos, constant.POS_TO_ID)
            ner = map_to_ids(stanford_ner, constant.NER_TO_ID)
            deprel = map_to_ids(stanford_deprel, constant.DEPREL_TO_ID)
            l = len(tokens)
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
            relation = constant.LABEL_TO_ID[d['relation']]
            processed += [{'tokens': tokens, 'pos': pos, 'ner': ner,
                           'deprel': deprel, 'subj_positions': subj_positions,
                           'obj_positions': obj_positions, 'relation': relation}]
            # processed += [(tokens, pos, ner, deprel, subj_positions, obj_positions, relation)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

    def __len__(self):
        # return 50
        return len(self.data)

    def next_batch(self):
        # continuously loop through dataset
        key = self.batch_index % len(self.data)
        batch = self.data[key]
        # print(batch.keys())
        batch_size = len(batch['tokens'])
        # batch = list(zip(*batch))
        # assert len(batch) == 7

        # sort all fields by lens for easy RNN operations
        # lens = [len(x) for x in batch[0]]
        # batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        batch['tokens'] = get_long_tensor(words, batch_size).cuda()
        batch['masks'] = torch.eq(batch['tokens'], 0).cuda()
        batch['pos'] = get_long_tensor(batch['pos'], batch_size).cuda()
        batch['ner'] = get_long_tensor(batch['ner'], batch_size).cuda()
        batch['deprel'] = get_long_tensor(batch['deprel'], batch_size).cuda()
        batch['subj_positions'] = get_long_tensor(batch['subj_positions'], batch_size).cuda()
        batch['obj_positions'] = get_long_tensor(batch['obj_positions'], batch_size).cuda()

        batch['relation'] = torch.LongTensor(batch['relation']).cuda()
        batch['orig_idx'] = list(range(batch_size))  # .cuda()
        self.batch_index += 1
        return batch

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0:
            raise IndexError
        if key >= len(self.data):
            key = key % len(self.data)
        # if key < 0 or key >= len(self.data):
        #     raise IndexError
        batch = self.data[key]
        #print(batch.keys())
        batch_size = len(batch['tokens'])
        #batch = list(zip(*batch))
        #assert len(batch) == 7

        # sort all fields by lens for easy RNN operations
        # lens = [len(x) for x in batch[0]]
        # batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        batch['tokens'] = get_long_tensor(batch['tokens'], batch_size).cuda()
        batch['masks'] = torch.eq(batch['tokens'], 0).cuda()
        batch['pos'] = get_long_tensor(batch['pos'], batch_size).cuda()
        batch['ner'] = get_long_tensor(batch['ner'], batch_size).cuda()
        batch['deprel'] = get_long_tensor(batch['deprel'], batch_size).cuda()
        batch['subj_positions'] = get_long_tensor(batch['subj_positions'], batch_size).cuda()
        batch['obj_positions'] = get_long_tensor(batch['obj_positions'], batch_size).cuda()

        batch['relation'] = torch.LongTensor(batch['relation']).cuda()
        batch['orig_idx'] = list(range(batch_size))#.cuda()

        return batch

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids


def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0] * (end_idx - start_idx + 1) + \
           list(range(1, length - end_idx))


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
                else x for x in tokens]

