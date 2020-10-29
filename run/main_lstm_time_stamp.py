import os

import torch

from data.toy_dataset import ToyDataset
from engine.lstm_engine import LSTMEngine
from models.lstm_model import LSTMModel
from utils.base_metric import write_metrics_to_txt
from utils.cuda import put_model_on_gpu, put_var_on_gpu
from utils.record import pickle_history


def main(cfg):
    engine = LSTMEngine(cfg)

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

    torch.save(model, cfg.save_model_path)
    pickle_history(engine.history, engine.log_path)


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
        default="config.lstm_time_stamp_cfg",
        help="The configuration file to use.")
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    main(cfg)
