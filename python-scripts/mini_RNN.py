import torch
from torch import nn

rnn = nn.GRU(1, 8)
input = torch.randn(5, 3, 1)
print(f'input ({input.shape}):\n{input}')
h0 = torch.randn(1, 3, 8)
output, hn = rnn(input, h0)

print(f'output ({output.shape}): \n{output}')
print(f'hn ({hn.shape}): \n{hn}')

print('')
print('Printing model weights:')
print(f'rnn.weight_ih_l0 ({rnn.weight_ih_l0.shape}):\n{rnn.weight_ih_l0}')
print('')
print(f'rnn.weight_hh_l0 ({rnn.weight_hh_l0.shape}):\n{rnn.weight_hh_l0}')
