import importlib

import torch

from dataset.toy_dataset import ToyDataset
from engine.lstm_engine import LSTMEngine
from models.lstm_model import LSTMModel
from utils.cuda import put_model_on_gpu


def main(args):
    cfg = importlib.import_module(args.config)
    log_dir = args.logdir
    engine = LSTMEngine(cfg, log_dir)

    dataset_val = ToyDataset(subset="val")
    dataloader_val = ToyDataset.get_dataloader(dataset_val, cfg.batch_size)

    model = LSTMModel(**cfg.lstm_cfg)
    model_weights_path = args.modelpath
    if model_weights_path != "":
        model_weights = torch.load(open(model_weights_path, "rb"))
        model.load_state_dict(model_weights)
    model = put_model_on_gpu(model, cfg.devices)

    loss_fn = cfg.loss_fn()
    optimizer = cfg.optimizer(model.parameters(), **cfg.optim_params)
    engine.compile(model, optimizer, loss_fn, cfg.metrics)

    _, metrics = engine.evaluate(dataloader_val)
    print(metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
        default="config.lstm_time_stamp_cfg",
        help="The configuration file to use.")
    parser.add_argument("--logdir", type=str,
        default="",
        help="The directory to save tensorboard logs and model.")
    parser.add_argument("--modelpath", type=str,
        default="",
        help="The trained PyTorch model weights path.")
    args = parser.parse_args()

    main(args)
