from .base_class import BaseClass
from .base_config import ConfigDict
from torch.utils.data import Dataset, DataLoader
import soundfile
import python_speech_features
from .base_vocab import Vocab
import numpy as np
import random
import torch as t
from .feature_utils import build_LFR_features
from .feature_utils import spec_augment
from pyvad import trim
import librosa


class BasePredumpParser(BaseClass):
    def __init__(self, config):
        super(BasePredumpParser, self).__init__()
        self.config = config
        self.load_vocab()

    def load_vocab(self):
        self.vocab = Vocab.load(self.config.vocab_path)

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
            F=20,
            T=17,
            pt=0.2,
            num_T=2,
            num_F=2,
            vocab_path='data/data_aishell/vocab.t'
        )
        return config

    def _load_wav(self, wav_file):
        sig, sample_rate = librosa.load(wav_file, sr=self.config.sample_rate)
        tmp = sig
        if self.config.use_vad:
            sig = trim(sig, sample_rate, fs_vad=self.config.sample_rate, hoplength=30, thr=0, vad_mode=2)
        if sig is None:
            return tmp
        else:
            return sig

    def _feature_mel(self, signal):
        tensor = python_speech_features.logfbank(
            signal,
            samplerate=self.config.sample_rate,
            winlen=self.config.window_size,
            winstep=self.config.window_hop,
            nfilt=self.config.n_mels,
            lowfreq=self.config.f_min,
            highfreq=self.config.f_max
        )
        # [L, H]
        return tensor

    def _feature_mfcc(self, signal):
        tensor = python_speech_features.mfcc(
            signal,
            samplerate=self.config.sample_rate,
            winlen=self.config.window_size,
            winstep=self.config.window_hop,
            nfilt=self.config.n_mels,
            lowfreq=self.config.f_min,
            highfreq=self.config.f_max
        )
        return tensor

    def _feature_delta(self, tensor):
        delta1 = python_speech_features.delta(tensor, self.config.delta_num)
        delta2 = python_speech_features.delta(delta1, self.config.delta_num)
        delta_feature = np.concatenate((tensor, delta1, delta2), -1)
        return delta_feature

    def _normalize(self, tensor):
        tensor = (tensor - tensor.mean()) / tensor.std()
        return tensor

    def _feature_lfr(self, tensor):
        feature = build_LFR_features(tensor, self.config.num_stack, self.config.num_skip)
        return feature

    def _aug_speed(self, sig, speed_rate):
        old_length = sig.shape[0]
        new_length = int(sig.shape[0] / speed_rate)
        old_indices = np.arange(old_length)
        new_indices = np.linspace(start=0, stop=old_length, num=new_length)
        nsig = np.interp(new_indices, old_indices, sig)
        return nsig

    def _spec_augment(self, tensor):
        if not isinstance(tensor, t.Tensor):
            tensor = t.from_numpy(tensor)

    def parse_wav(self, path):
        raise NotImplementedError

    def parse_word(self, str):
        raise NotImplementedError

    def build_dataset(self):
        raise NotImplementedError

    def build_iters(self):
        raise NotImplementedError

    @property
    def collate_fn(self):
        def collate_fn(batch):
            return batch
        return collate_fn

