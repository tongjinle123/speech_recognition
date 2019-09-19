import torch as t
import librosa
from torch.utils.data import DataLoader, Dataset
import os


class DS(Dataset):
    def __init__(self):
        super(DS, self).__init__()
        folder = 'data/data_aishell/wav/train/S0002/'
        self.files = [folder + i for i in os.listdir(folder)]

    def __len__(self):

        return len(self.files)

    def __getitem__(self, item):
        return librosa.load(self.files[item])


class Prefetcher(DataLoader):
    def __init__(self, *args, **kwargs):
        super(Prefetcher, self).__init__(*args, **kwargs)

    def preload(self):
        try:
            self.next_data = next(self)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

    def __next__(self):
        print('next')

    def __getitem__(self, item):
        print('getitem')



if __name__ == '__main__':
    ds = DS()
    dl = Prefetcher(ds, batch_size=32, num_workers=10)


class Prefetcher(DataLoader):
    def __init__(self, *args, **kwargs):
        super(Prefetcher, self).__init__(*args, **kwargs)
        self.next_data = self.next()

    def preload(self):
        try:
            self.next_data = next(self)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

    def __iter__(self):
        print('iter')