import Levenshtein as Lev


def calculate_cer(s1, s2, eos_token='&'):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence (hyp)
        s2 (string): space-separated sentence (gold)
    """

    s2 = s2.split(' ')
    s2 = [i for i in s2 if i != eos_token]
    s2 = ''.join(s2)
    s1 = s1.split(' ')
    s1 = [i for i in s1 if i != eos_token]
    s1 = ''.join(s1)
    word_num = len(s2)
    return Lev.distance(s1, s2) / word_num


def calculate_cer_ctc(pre, tgt):
    #s1 predicted, s2 target
    tgt = tgt.split(' ')
    tgt = ''.join(tgt)
    pre = pre.split(' ')
    pre = ''.join(pre)
    word_num = len(pre)
    return Lev.distance(pre, tgt) / word_num