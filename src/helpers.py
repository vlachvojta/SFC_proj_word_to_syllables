import Levenshtein as lev


def levenstein_loss(a, b):
    if isinstance(a, list):
        a = ''.join(a)
    if isinstance(b, list):
        b = ''.join(b)
    return 100.0 * lev.distance(a, b) / len(b)