# coding: utf-8

import torch
from torch.autograd import Variable

# no_grad was introduced in torch 0.4, so we need to check whether it is available in case we need to fall back to older
# API for creating variables without gradients. Do it once at import time and cache the result,
_TORCH_HAS_NO_GRAD = 'no_grad' in dir(torch)


def cuda_variable(x, volatile=False):
    if _TORCH_HAS_NO_GRAD:
        if volatile:
            with torch.no_grad():
                v = Variable(x)
        else:
            v = Variable(x)
    else:
        v = Variable(x, volatile=volatile)

    return v.cuda()
