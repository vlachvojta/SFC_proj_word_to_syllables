import Levenshtein as lev
import torch


def levenstein_loss(a, b):
    if isinstance(a, list):
        a = ''.join(a)
    if isinstance(b, list):
        b = ''.join(b)
    return 100.0 * lev.distance(a, b) / len(b)


def transpose(data: torch.Tensor):
    if data.shape[-1] == 1:
        raise ValueError(f'Tensor already transposed. data.shape: {data.shape}')

    new_shape = [dim for dim in data.shape] + [1]
    data = torch.reshape(data, new_shape)
    return data


def de_transpose(data: torch.Tensor):
    if not data.shape[-1] == 1:
        raise ValueError(f'Cannot de_transpose tensor. data.shape: {data.shape}')

    new_shape = [dim for dim in data.shape[:-1]]
    data = torch.reshape(data, new_shape)
    return data
