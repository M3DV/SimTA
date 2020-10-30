import torch.nn as nn
import torch.optim as optim

from utils.mean_squared_error import MeanSquaredError


# GPU config
devices = [0]

# data config
batch_size = 128

# model config
lstm_cfg = {
    "d_in": 1,
    "d_hidden": 256,
    "d_out": 3,
    "num_layers": 6,
    "use_tau": False,
    "use_time_stamp": False
}

# training config
max_lr = 1e-3
optimizer = optim.AdamW
optim_params = {
    "lr": max_lr
}
loss_fn = nn.MSELoss
lr_scheduler = optim.lr_scheduler.OneCycleLR
lr_scheduler_params = {
    "max_lr": max_lr
}
epochs = 10
metrics = [MeanSquaredError()]

# save model path
save_model_path = "trained_models/lstm/model.pt"
