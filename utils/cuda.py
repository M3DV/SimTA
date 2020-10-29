import os
import pickle

import torch.nn as nn


def put_var_on_gpu(var, devices, requires_grad=False):
    if len(devices) == 1:
        var = var.cuda(devices[0])
    return var


def put_model_on_gpu(model, devices):
    if len(devices) == 1:
        model = model.cuda(devices[0])
    else:
        model = nn.DataParallel(model, device_ids=devices)
    return model
