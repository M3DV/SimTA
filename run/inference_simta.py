import importlib
import os

import numpy as np
import torch
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

from dataset.toy_dataset import ToyDataset
from engine.simta_engine import SimTAEngine
from models.simta import SimTANet
from toy_data.toy_data import ToyData
from utils.cuda import put_model_on_gpu, put_var_on_gpu
from utils.base_metric import write_metrics_to_txt


def main(cfg):
    engine = SimTAEngine(cfg)
    
    dataset_val = ToyDataset(subset="val")
    toy_data_params = dataset_val.params
    toy_data_gen = ToyData(*toy_data_params)
    dataloader_val = ToyDataset.get_dataloader(dataset_val, cfg.batch_size)

    model = put_model_on_gpu(torch.load(cfg.save_model_path), cfg.devices)

    loss_fn = cfg.loss_fn()
    optimizer = cfg.optimizer(model.parameters(), **cfg.optim_params)
    engine.compile(model, optimizer, loss_fn, cfg.metrics)
    
    idx = 16
    marker_size = 400
    title_size = 70
    legend_size = 20
    tick_size = 20
    label_size = 50
    font = fm.FontProperties(
        fname="/home/kaiming/Downloads/times-new-roman.ttf")
    for _, sample in enumerate(dataloader_val):
        t, x, y, y_true = sample
        t_series = np.linspace(t[idx, 0], t[idx].sum() + 4, 10000)
        y_series = toy_data_gen.generate_denoised_series(t_series)
        plt.plot(t_series, y_series, "gray", dashes=(5, 2, 1, 2),
            label="Denoised Ground Truth", lw=3)
        plt.scatter(t[idx].cumsum(dim=0), x[idx], s=marker_size, marker="v",
            label="Noisy Asynchronous Input")
        x = put_var_on_gpu(x, cfg.devices)
        t = put_var_on_gpu(t, cfg.devices)
        output = model(x, t)
        plt.scatter(np.linspace(t[idx].detach().cpu().sum() + 1,
            t[idx].detach().cpu().sum() + 3, 3),
            output.detach().cpu().numpy()[idx], s=marker_size, marker="d",
            label="Prediction")
        plt.legend(fontsize=legend_size, loc="upper right")
        plt.xticks(fontsize=tick_size)
        plt.yticks(fontsize=tick_size)
        plt.title("Synthetic Time Series", fontproperties=font,
            fontsize=title_size)
        plt.xlabel("Time Steps", fontproperties=font, fontsize=label_size)
        plt.show()

        break


if __name__ == "__main__":
    import argparse
    import importlib

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
        default="config.simta_cfg",
        help="The configuration file to use.")
    args = parser.parse_args()
    cfg = importlib.import_module(args.config)

    main(cfg)
