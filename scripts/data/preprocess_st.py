import os
import tarfile
from tqdm import tqdm
import torch as t
import shutil
import pandas as pd
import sys
sys.path.append(os.getcwd())
from src.base.base_vocab import Vocab
import soundfile
from src.base.base_vocab import Vocab

vocab = Vocab.load('data/data_aishell/vocab.t')
v = list(vocab._token2id.keys())


def untar(file_from, folder_to):
    file = tarfile.open(file_from)
    file.extractall(folder_to)
    file.close()


def untar_st(file_from, folder_to):
    untar(file_from, folder_to)

# def untar_aishell(file_from, folder_to):
#     untar(file_from, folder_to)
#     folder = file_from[:-4] + '/wav/'
#     for i in tqdm(os.listdir(folder)):
#         untar(folder + i, folder)
#         os.remove(os.path.join(folder, i))
#     print(f'untar done')


def cal_duration(file):
    prefix = 'data/ST-CMDS-20170001_1-OS/'
    sig, sr = soundfile.read(prefix+file)
    duration = len(sig) / sr
    return duration


def get_text(file):
    prefix = 'data/ST-CMDS-20170001_1-OS/'
    with open(prefix+file, encoding='utf8') as reader:
        data = reader.readline()
    return data


def if_filter(s):
    for i in s:
        if i not in v:
            return 1
    return 0


def build_manifist_st(data_folder, manifist_folder):
    file_list = os.listdir(data_folder)
    waves = {i.split('.')[0]: i for i in file_list if i.endswith('.wav')}
    txts = {i.split('.')[0]: i for i in file_list if i.endswith('.txt')}
    wave_df = pd.DataFrame.from_dict(waves, orient='index', columns=['wave_file'])
    txt_df = pd.DataFrame.from_dict(txts, orient='index', columns=['txt_file'])
    df = pd.merge(wave_df, txt_df, left_on=wave_df.index, right_on=txt_df.index)
    df['target'] = df['txt_file'].apply(get_text)
    df['duration'] = df['wave_file'].apply(cal_duration)
    del df['txt_file']
    del df['key_0']
    df['wave_file'] = 'data/ST-CMDS-20170001_1-OS/' + df['wave_file']
    df['if_filter'] = df['target'].apply(if_filter)
    df = df[df['if_filter'] == 0]
    samples = [i[1] for i in df.to_dict('index').items() if len(i[1]['target']) < 1]
    t.save(samples, os.path.join(manifist_folder, 'st.manifist'))


# def build_vocab(train_manifist_file, vocab_path):
#     vocab = Vocab()
#     manifist = t.load(train_manifist_file)
#     for sample in manifist:
#         vocab.consume_sentance(sample['target'])
#     vocab.build()
#     vocab.save(vocab_path)


if __name__ == '__main__':
    untar_st('data/ST-CMDS-20170001_1-OS.tar.gz', 'data/')
    build_manifist_st('data/ST-CMDS-20170001_1-OS', 'data/ST-CMDS-20170001_1-OS')
