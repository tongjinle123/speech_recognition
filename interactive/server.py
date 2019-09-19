import sys
import os
sys.path.append(os.getcwd())

from flask import Flask, request
from main import get_model
import torch as t
import json
from flask_cors import CORS, cross_origin

print("Loading model...")

parser, model = get_model('20190724_1431_49_360_675000', 'Solver')
model.eval()
model.init_beam_decoder()
print("Model loaded")

app = Flask(__name__)


@app.route("/recognize", methods=["POST"])
@cross_origin(origin='http://172.18.34.25', headers=['Content-Type'])
def recognize():
	
    f = request.files["file"]
    print(f)
    f.save("test.wav")
    with t.no_grad():
        feature, length = parser.parser_wav_inference('test.wav')
        output = model.beam_decode_feature(feature.float().cuda(), length.cuda())

    return ''.join(output)


if __name__ == '__main__':
    app.run("0.0.0.0", debug=True, port=5001)

