import torch as t
from collections import Counter


def tokenize_fn(x):
    return [i for i in x]


class Vocab:

    def __init__(self, PAD = '$', UNK = '%', BOS = '^', EOS = '&',
                 tokenize_fn=tokenize_fn):
        self._counter = Counter()
        self.PAD = PAD
        self.UNK = UNK
        self.BOS = BOS
        self.EOS = EOS
        self._token2id = {v: i for i, v in enumerate([PAD, UNK, BOS, EOS]) if v is not None}
        self._id2token = None
        self._tokenize_fn = tokenize_fn

    def consume_sentance(self, sentance: str):
        sentance = self._tokenize_fn(sentance)
        self._counter.update(sentance)

    def consume_sentance_list(self, sentance_list: list):
        for sentance in sentance_list:
            self.consume_sentance(sentance)

    def build(self, min_count: int = 1, max_vocab: int = 20000):
        for i in self._counter.most_common(max_vocab):
            if i[1] >= min_count:
                self._token2id[i[0]] = len(self._token2id)
        self._id2token = [i for i in self._token2id]
        print(f'total {len(self._token2id)} words in vocab')

    def save(self, path: str):
        assert self._id2token is not None
        all = (self._id2token, self._token2id, self.PAD, self.UNK, self.BOS, self.EOS)
        t.save(all, path)
        print(f'vocab saved in {path}')

    @classmethod
    def load(cls, path: str):
        obj = cls()
        all = t.load(path)
        obj._id2token = all[0]
        obj._token2id = all[1]
        obj.PAD = all[2]
        obj.UNK = all[3]
        obj.BOS = all[4]
        obj.EOS = all[5]
        print(f'vocab loaded from {path}')
        return obj

    def convert_str(self, string:str, use_bos: bool = True, use_eos: bool = True):
        token = self._tokenize_fn(string)
        if use_bos:
            token = [self.BOS] + token
        if use_eos:
            token = token + [self.EOS]
        id = self.convert_token(token)
        return id

    def convert_token(self, token: list):
        id = [self._token2id.get(i, self._token2id[self.UNK]) for i in token]
        return id

    def convert_id(self, id: list):
        assert self._id2token is not None
        token = [self._id2token[i] for i in id]
        # if use_label:
        #     token = [self.BOS] + token + [self.EOS]
        return token

    def convert_id2str(self, id: list):
        assert self._id2token is not None
        token = [self._id2token[i] for i in id if i!=self._token2id[self.PAD]]
        token = ' '.join(token)
        return token

    @property
    def bos_id(self):
        return self._token2id[self.BOS]

    @property
    def eos_id(self):
        return self._token2id[self.EOS]

    @property
    def pad_id(self):
        return self._token2id[self.PAD]

    @property
    def vocab_size(self):
        return len(self._token2id)
