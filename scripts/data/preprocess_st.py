import os
import tarfile
from tqdm import tqdm
import torch as t
import shutil
import sys
sys.path.append(os.getcwd())
from src.base.base_vocab import Vocab
import soundfile


def untar(file_from, folder_to):
    file = tarfile.open(file_from)
    file.extractall(folder_to)
    file.close()


def untar_st(file_from, folder_to):
    pass

# def untar_aishell(file_from, folder_to):
#     untar(file_from, folder_to)
#     folder = file_from[:-4] + '/wav/'
#     for i in tqdm(os.listdir(folder)):
#         untar(folder + i, folder)
#         os.remove(os.path.join(folder, i))
#     print(f'untar done')


def cal_duration(file):
    sig, sr = soundfile.read(file)
    duration = len(sig) / sr
    return duration


def build_manifist_aishell(data_folder, manifist_folder):
    with open(os.path.join(data_folder, 'transcript', 'aishell_transcript_v0.8.txt'), encoding='utf8') as file:
        transcripts = file.readlines()
        transcripts_dict = {i.strip().split(' ')[0]: ''.join(i.strip().split(' ')[1:]) for i in transcripts}
    wave_folder = os.path.join(data_folder, 'wav')
    data_sets = ['test', 'dev', 'train']
    for data_set in data_sets:
        manifist = []
        people_list = os.listdir(os.path.join(wave_folder, data_set))
        for people in people_list:
            files = os.listdir(os.path.join(wave_folder, data_set, people))
            for file in files:
                try:
                    wave_file = os.path.join(os.path.join(wave_folder, data_set, people, file))
                    sample = {'wave_file': wave_file,
                              'target': transcripts_dict[file[:-4]],
                              'duration': cal_duration(wave_file)}
                    manifist.append(sample)
                except:
                    pass
        print(os.path.join(manifist_folder, data_set + '.manifist'))
        t.save(manifist, os.path.join(manifist_folder, data_set + '.manifist'))
    print('manifist built')


def build_vocab(train_manifist_file, vocab_path):
    vocab = Vocab()
    manifist = t.load(train_manifist_file)
    for sample in manifist:
        vocab.consume_sentance(sample['target'])
    vocab.build()
    vocab.save(vocab_path)