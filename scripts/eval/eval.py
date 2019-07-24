from main import get_model
import torch as t
from ctcdecode import CTCBeamDecoder


parser, model = get_model('20190717_1039_29_411_770000', 'WSolver')
parser.config.batch_size=32
train,dev,test = parser.build_iters()
from tqdm import tqdm
from src.models.utils.score import calculate_cer_ctc

model.cpu()
model.eval()
print('-')
model.init_beam_decoder(num_processes=16, beam_width=32)
import numpy as np
scores = []
for i in tqdm(test):
    tgt = i['tgt']
    tgt = [model.vocab.convert_id2str(i) for i in tgt]
    model_output = model.beam_decode(i)
    ss = [calculate_cer_ctc(i[0], i[1]) for i in zip(model_output, tgt)]
    score = np.mean(ss)
    print(score)
    scores.append(score)
    print('current mean', np.mean(scores))
import torch as t
t.save(scores, 'test32.t')
print(np.mean(scores))
