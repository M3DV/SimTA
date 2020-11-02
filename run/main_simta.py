import importlib
import os

import torch

from dataset.toy_dataset import ToyDataset
from engine.simta_engine import SimTAEngine
from models.simta import SimTANet
from utils.cuda import put_model_on_gpu
from utils.record import pickle_history


def main(args):
    cfg = importlib.import_module(args.config)
    log_dir = args.logdir
    engine = SimTAEngine(cfg, log_dir)

    dataset_train = ToyDataset(subset="train")
    dataloader_train = ToyDataset.get_dataloader(dataset_train,
        cfg.batch_size)
    dataset_val = ToyDataset(subset="val")
    dataloader_val = ToyDataset.get_dataloader(dataset_val, cfg.batch_size)

    model = put_model_on_gpu(SimTANet(**cfg.simta_cfg), cfg.devices)

    loss_fn = cfg.loss_fn()
    optimizer = cfg.optimizer(model.parameters(), **cfg.optim_params)
    engine.compile(model, optimizer, loss_fn, cfg.metrics)
    scheduler = cfg.lr_scheduler(engine.optimizer,
        total_steps=cfg.epochs * len(dataloader_train),
        **cfg.lr_scheduler_params)

    engine.train(dataloader_train, cfg.epochs, dataloader_val, scheduler)

    torch.save(model.state_dict(),
        os.path.join(engine.log_path, "model_weights.pth"))
    pickle_history(engine.history, engine.log_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
        default="config.simta_cfg",
        help="The configuration file to use.")
    parser.add_argument("--logdir", type=str,
        required=True,
        help="The directory to save tensorboard logs and model.")
    args = parser.parse_args()

    main(args)
