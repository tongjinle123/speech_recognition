from src.base import BaseParser
from src.base import ConfigDict
import torch as t
from torch.utils.data import Dataset, DataLoader
from .utils import Padder


class ParserAishell(BaseParser):
    def __init__(self, config):
        super(ParserAishell, self).__init__(config)


    @classmethod
    def load_default_config(cls) -> ConfigDict:
        config = ConfigDict()
        config.add(
            batch_size=64,
            sample_rate=16000,
            n_mels=80,
            window_size=0.025,
            window_hop=0.01,
            f_min=40,
            f_max=8000,
            num_stack=4,
            num_skip=3,
            delta_num=2,
            speed_min=0.9,
            speed_max=1.1,
            use_vad=True,
            num_worker=16,
            F=27,
            T=25,
            pt=0.2,
            num_T=1,
            num_F=2,
            train_max_duration=10,
            vocab_path='data/data_aishell/vocab.t'
        )
        input_size = config.n_mels * config.num_stack
        config.add(input_size=input_size)
        return config

    def parse_wav(self, path, if_augment=False):
        sig = self._load_wav(path)
        if if_augment:
            sig = self._aug_speed(sig)

        feature = self._feature_mel(sig)
        feature = self._feature_lfr(feature)
        feature = self._normalize(feature)
        if if_augment:
            feature = self._aug_freq_time_mask(feature)
        feature = t.from_numpy(feature)
        return feature

    def parse_word(self, str):
        id = self.vocab.convert_str(str, False, False)
        return id

    def _build_iter(self, manifist_file, if_augment, if_filter_duration=False):
        max_duration=None if not if_filter_duration else self.config.train_max_duration
        dataset = AishellDataSet(manifist_file, if_augment=if_augment, max_duration=max_duration, parser=self)
        dataloader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_worker,
            collate_fn=collate_fn, drop_last=True, pin_memory=False)
        return dataloader

    def parser_wav_inference(self, path):
        feature = self.parse_wav(path, if_augment=False)
        feature = feature.unsqueeze(0)
        length = t.LongTensor([feature.size(1)])
        return feature, length

    def build_iters(self):
        train_iter = self._build_iter('data/data_aishell/train.manifist', if_augment=True, if_filter_duration=True)
        dev_iter = self._build_iter('data/data_aishell/dev.manifist', if_augment=False, if_filter_duration=False)
        test_iter = self._build_iter('data/data_aishell/test.manifist', if_augment=False, if_filter_duration=False)
        return train_iter, dev_iter, test_iter


class AishellDataSet(Dataset):
    def __init__(self, manifist_file, parser, max_duration=None, if_augment=False):
        super(AishellDataSet, self).__init__()
        self.datas = t.load(manifist_file)
        self.parser = parser
        self.if_augment = if_augment
        if max_duration is not None:
            self.datas = [i for i in self.datas if i['duration'] <= max_duration]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        sample = self.datas[item]
        feature = self.parser.parse_wav(sample['wave_file'], if_augment=self.if_augment)
        target = self.parser.parse_word(sample['target'])
        return feature, target


def collate_fn(batch):
    features = [i[0] for i in batch]
    tgts = [i[1] for i in batch]
    features, feature_len = Padder.pad_tri(features, 0)
    tgts, tgt_len = Padder.pad_two(tgts, 0)
    return {'wave': features, 'wave_len': t.LongTensor(feature_len), 'tgt': tgts.long(), 'tgt_len': t.LongTensor(tgt_len)}
