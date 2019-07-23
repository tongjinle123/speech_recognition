import torch as t


class Padder:

    @staticmethod
    def pad_two(inputs, pad_value, lengths=None):
        if lengths is None:
            lengths = [len(i) for i in inputs]
        batch_size = len(inputs)
        output = t.full((batch_size, max(lengths)), pad_value)
        for index, (i, l) in enumerate(zip(inputs, lengths)):
            if isinstance(i, list):
                n = t.Tensor(i)
            else:
                n = i
            output[index, :l] = n
        return output, lengths

    @staticmethod
    def pad_tri(inputs, pad_value, lengths=None):
        if lengths is None:
            lengths = [len(i) for i in inputs]
        batch_size = len(inputs)
        output = t.full((batch_size, max(lengths), inputs[0].size(-1)), pad_value)
        for index, (i, l) in enumerate(zip(inputs, lengths)):
            output[index, :l, :] = i
        return output, lengths