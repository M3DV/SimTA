import importlib
import os

import torch

from dataset.toy_dataset import ToyDataset
from engine.lstm_engine import LSTMEngine
from models.lstm_model import LSTMModel
from utils.cuda import put_model_on_gpu
from utils.record import pickle_history


def main(args):
    cfg = importlib.import_module(args.config)
    log_dir = args.logdir
    engine = LSTMEngine(cfg, log_dir)

    dataset_train = ToyDataset(subset="train")
    dataloader_train = ToyDataset.get_dataloader(dataset_train,
        cfg.batch_size)
    dataset_val = ToyDataset(subset="val")
    dataloader_val = ToyDataset.get_dataloader(dataset_val, cfg.batch_size)

    model = put_model_on_gpu(LSTMModel(**cfg.lstm_cfg), cfg.devices)

    loss_fn = cfg.loss_fn()
    optimizer = cfg.optimizer(model.parameters(), **cfg.optim_params)
    engine.compile(model, optimizer, loss_fn, cfg.metrics)
    scheduler = cfg.lr_scheduler(engine.optimizer,
        total_steps=cfg.epochs * len(dataloader_train),
        **cfg.lr_scheduler_params)

    engine.train(dataloader_train, cfg.epochs, dataloader_val, scheduler)

    torch.save(model, os.path.join(engine.log_path, "model.pth"))
    pickle_history(engine.history, engine.log_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
        default="config.lstm_time_interval_cfg",
        help="The configuration file to use.")
    parser.add_argument("--logdir", type=str,
        required=True,
        help="The directory to save tensorboard logs and model.")
    args = parser.parse_args()

    main(args)
