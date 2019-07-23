import sys
import os
sys.path.append(os.getcwd())
from main import get_model

parser, model = get_model('20190717_1039_29_397_745000', 'Solver')
parser.config.batch_size=32
train,dev,test = parser.build_iters()
for i in test:
    break
model.init_beam_decoder(num_processes=16, beam_width=32)
model_output = model.beam_decode(i)

