import torch as t




def pad_list(xs, pad_value):
    # From: espnet/src/nets/e2e_asr_th.py: pad_list()
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, * xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


class Masker:
    @staticmethod
    def get_pad_mask(padded_input, input_lengths=None, pad_idx=0):
        """
        padding position is set to 0, either use input_lengths or pad_idx
        """
        assert input_lengths is not None or pad_idx is not None
        if input_lengths is not None:
            # padded_input: N x T x ..
            N = padded_input.size(0)
            pad_mask = padded_input.new_zeros(padded_input.size())  # B x T
            for i in range(N):
                pad_mask[i, input_lengths[i]:] = 1
        if pad_idx is not None:
            # padded_input: N x T
            assert padded_input.dim() == 2
            pad_mask = padded_input.ne(pad_idx).float()
        # unsqueeze(-1) for broadcast
        return pad_mask

    @staticmethod
    def get_dot_attention_mask(query_pad_mask, key_pad_mask):
        dot_attention_mask = t.bmm(query_pad_mask.unsqueeze(-1), key_pad_mask.unsqueeze(-1).transpose(-1, -2))
        return dot_attention_mask

    @staticmethod
    def get_subsequent_mask(seq):
        ''' For masking out the subsequent info. '''

        sz_b, len_s = seq.size()
        subsequent_mask = t.tril(
            t.ones((len_s, len_s), device=seq.device, dtype=t.uint8), diagonal=0)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
        return subsequent_mask.float()

if __name__ == '__main__':


    query = t.Tensor([[1,2,3],[2,3,0]])
    query_len = t.Tensor([3,2]).long()
    key = t.Tensor([[1,2,3,4,5,6],[2,3,4,5,0,0]])
    print(Masker.get_pad_mask(query, query_len))
    query_pad_mask = Masker.get_pad_mask(query)
    key_pad_mask = Masker.get_pad_mask(key)
    print(Masker.get_dot_attention_mask(query_pad_mask, query_pad_mask))
    self_attention_mask = Masker.get_dot_attention_mask(query_pad_mask, query_pad_mask)
    dot_attention_mask = Masker.get_dot_attention_mask(query_pad_mask, key_pad_mask)
    print(Masker.get_subsequent_mask(query))
    sub_mask = Masker.get_subsequent_mask(query)
    se = self_attention_mask * sub_mask
    print(se)