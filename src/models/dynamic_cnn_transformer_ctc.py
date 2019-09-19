from src.base import ConfigDict
from src.base import BaseModel
import fairseq as fs
import torch as t
from src.base.utils import Masker
from .utils.score import calculate_cer_ctc
from src.base.utils import Pack
from ctcdecode import CTCBeamDecoder


class DynamicCNNTransformerCTC(BaseModel):
    def __init__(self, config, vocab):
        super(DynamicCNNTransformerCTC, self).__init__(config=config)
        self.vocab = vocab
        self.input_layer = InputLayer(config.input_size, config.model_size, config.dropout)
        self.encoder = DynamicCnnTransformerEncoder(
            input_size=config.model_size, conv_size=config.conv_size, kernel_size_list=config.kernel_size_list,
            dropout=config.dropout, num_head=config.num_head, ff_size=config.ff_size
        )
        self.output_linear = t.nn.Linear(config.model_size, vocab.vocab_size)
        t.nn.init.xavier_normal_(self.output_linear.weight)

    @classmethod
    def load_default_config(cls):
        config = ConfigDict()
        config.add(
            model_size=512,
            conv_size=64,
            ff_size=1024,
            dropout=0.1,
            num_head=8,
            kernel_size_list=[3, 5, 7, 11, 31, 31, 31],
        )
        return config

    def init_beam_decoder(self, alpha=0.8, beta=0.3, cutoff_top_n=40, cutoff_prob=1.0, beam_width=32, num_processes=4, use_lm=True):
        lm_path = "lm/zh_giga.no_cna_cmn.prune01244.klm" if use_lm else None
        blank_index = 1
        self.beam_decoder = CTCBeamDecoder(
            labels=self.vocab._id2token, model_path=lm_path, alpha=alpha, beta=beta, cutoff_top_n=cutoff_top_n,
            cutoff_prob=cutoff_prob, beam_width=beam_width, num_processes=num_processes, blank_id=blank_index,
            log_probs_input=True
        )

    def beam_decode(self, input):
        model_output, output_len = self.forward(input)
        assert self.beam_decoder
        model_output = t.nn.functional.log_softmax(model_output, -1)
        out, score, offset, out_len = self.beam_decoder.decode(model_output, output_len)
        output_tokens = [self.vocab.convert_id2str(i[0][:v[0]]) for i, v in zip(out, out_len)]
        return output_tokens

    def beam_decode_feature(self, feature, feature_len):
        input = {'wave':feature, 'wave_len':feature_len, 'tgt': None, 'tgt_len':None}
        output_tokens = self.beam_decode(input)
        return output_tokens

    def forward(self, input):
        wave, wave_len, text, text_len = input['wave'], input['wave_len'], input['tgt'], input['tgt_len']
        net = self.input_layer(wave)
        wave_pad_mask = Masker.get_pad_mask(wave[:, :, 0], wave_len).unsqueeze(-1)
        net = self.encoder(net, wave_pad_mask)
        net = self.output_linear(net)
        return net, input['wave_len']

    def decode(self, yp, yp_lens):
        idxs = yp.argmax(2)
        texts = []
        for idx, out_len in zip(idxs, yp_lens):
            idx = idx[:out_len]
            text = ""
            last = None
            for i in idx:
                if i.item() not in (last, 1):
                    text += self.vocab.convert_id([i.item()])[0]
                last = i
            texts.append(text)
        return texts

    def iterate(self, input, optimizer=None, is_train=True):
        output, output_len = self.forward(input)

        metrics = self.cal_metrics(output, output_len, input['tgt'], input['tgt_len'])
        if optimizer is not None and is_train and metrics.loss!=-100:
            optimizer.zero_grad()
            metrics.loss.backward()
            t.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            optimizer.step()
        return metrics, None

    def cal_metrics(self, output, output_len, tgt, tgt_len):
        pack = Pack()
        _loss = t.nn.CTCLoss(1)
        log_prob = t.nn.functional.log_softmax(output, -1).transpose(0, 1)
        # t b c
        loss = _loss(log_prob, tgt, output_len, tgt_len)
        if t.isinf(loss):
            loss = -100
            cer = 100
            pack.add(loss=loss, cer=t.Tensor([cer]))
            print('inf loss skiped')
            return pack
        else:
            output_str = self.decode(output, output_len)
            tgt_str = [self.vocab.convert_id2str(i) for i in tgt]
            cer = sum([calculate_cer_ctc(i[0], i[1]) for i in zip(output_str, tgt_str)]) * 100 / len(output_str)
            pack.add(loss=loss, cer=t.Tensor([cer]))
            return pack


class InputLayer(t.nn.Module):
    def __init__(self, input_size, model_size, dropout):
        super(InputLayer, self).__init__()
        self.input_linear = t.nn.Linear(input_size, model_size)
        self.input_layer_norm = t.nn.LayerNorm(model_size)
        self.input_highway = fs.modules.Highway(input_dim=model_size, num_layers=2)
        self.input_dropout = t.nn.Dropout(dropout)
        t.nn.init.xavier_normal_(self.input_linear.weight)

    def forward(self, wave_feature):
        net = self.input_linear(wave_feature)
        net = self.input_dropout(net)
        net = self.input_layer_norm(net)
        net = self.input_highway(net)
        return net


class DynamicCnnTransformerEncoder(t.nn.Module):
    def __init__(self, input_size, conv_size, kernel_size_list, dropout, num_head, ff_size,
                 weight_softmax=True, renorm_padding=True):
        super(DynamicCnnTransformerEncoder, self).__init__()
        self.layers = t.nn.ModuleList(
            [DynamicCnnEncoderLayer(
                input_size=input_size, conv_size=conv_size, kernel_size=kernel_size_list[i], dropout=dropout,
                num_head=num_head, ff_size=ff_size, weight_softmax=weight_softmax, renorm_padding=renorm_padding
            ) for i in range(len(kernel_size_list))]
        )

    def forward(self, input, pad_mask):
        for layer in self.layers:
            input = layer(input, pad_mask)
        return input


class DynamicCnnEncoderLayer(t.nn.Module):
    def __init__(self, input_size, conv_size, kernel_size, dropout, num_head, ff_size,
                 weight_softmax=True, renorm_padding=True):
        super(DynamicCnnEncoderLayer, self).__init__()
        self.cnn_block = DynamicCnnBlock(
            input_size, conv_size, kernel_size, dropout, num_head, weight_softmax=weight_softmax,
            renorm_padding=renorm_padding
        )
        self.feed_forward_block = FeedForwardBlock(input_size, ff_size, dropout, type='linear')

    def forward(self, input, pad_mask):
        net = self.cnn_block(input, pad_mask)
        net = self.feed_forward_block(net)
        return net


class DynamicCnnBlock(t.nn.Module):
    def __init__(self, input_size, conv_size, kernel_size, dropout, num_head, weight_softmax=True, renorm_padding=True):
        super(DynamicCnnBlock, self).__init__()
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        self.input_linear = t.nn.Sequential(
            t.nn.Linear(input_size, 2 * conv_size),
            t.nn.GLU(-1)
        )

        self.conv = fs.modules.DynamicConv1dTBC(
            input_size=conv_size, kernel_size=kernel_size, padding_l=padding_l, num_heads=num_head,
            weight_dropout=dropout, weight_softmax=weight_softmax, renorm_padding=renorm_padding
        )
        self.output_linear = t.nn.Linear(conv_size, input_size)
        self.dropout = t.nn.Dropout(dropout)
        self.layer_norm = t.nn.LayerNorm(input_size)
        t.nn.init.xavier_normal_(self.input_linear[0].weight)
        t.nn.init.xavier_normal_(self.output_linear.weight)

    def forward(self, input, pad_mask):
        #padmask B, L, 1
        # B, L, H
        res = input
        net = self.input_linear(input)
        # B, L, H
        net = net.masked_fill(pad_mask==0, 0)

        net = net.transpose(0, 1).contiguous()
        net = self.conv(net)
        net = net.transpose(0, 1).contiguous()
        net = self.output_linear(net)
        net = self.dropout(net)
        net += res
        net = self.layer_norm(net)
        return net


class FeedForwardBlock(t.nn.Module):
    def __init__(self, input_size, ff_size, dropout, type='linear'):
        super(FeedForwardBlock, self).__init__()
        self.core_type = type
        if type == 'linear':
            self.core = t.nn.Sequential(
                t.nn.Linear(input_size, ff_size),
                t.nn.ReLU(),
                t.nn.Dropout(dropout),
                t.nn.Linear(ff_size, input_size)
            )
            t.nn.init.xavier_normal_(self.core[0].weight)
            t.nn.init.xavier_normal_(self.core[-1].weight)
        elif type == 'conv':
            self.core = t.nn.Sequential(
                t.nn.Conv1d(input_size, ff_size, kernel_size=1),
                t.nn.ReLU(),
                t.nn.Dropout(dropout),
                t.nn.Linear(ff_size, input_size)
            )
            t.nn.init.kaiming_normal_(self.core[0].weight)
            t.nn.init.kaiming_normal_(self.core[-1].weight)

        else:
            raise ValueError
        self.dropout = t.nn.Dropout(dropout)
        self.layer_norm = t.nn.LayerNorm(input_size)

    def forward(self, input):
        #B, L, H
        res = input
        if not self.core_type == 'linear':
            input = input.transpose(1, 2)
            # B, H, L
            net = self.core(input)
            # B, H, L
            net = net.transpose(1, 2)
            # B, L, H
        else:
            net = self.core(input)

        net = self.dropout(net)
        net += res
        net = self.layer_norm(net)
        return net
