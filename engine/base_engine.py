import os
import shutil
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class BaseEngine:

    def __init__(self, cfg, log_dir):
        self.cfg = cfg
        start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_path = os.path.join(log_dir, start_time)
        self.cfg_path = os.path.join(self.log_path, "config")
        self.history = {"train": {}, "val": {}}

    def compile(self, model, optimizer, loss_fn, metrics=[]):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics

    def forward(self, sample):
        raise NotImplementedError("forward method not implemented.")

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def _metrics_epoch_begin(self):
        for i in range(len(self.metrics)):
            self.metrics[i].on_epoch_begin()

    def _metrics_batch_end(self, output, target):
        for i in range(len(self.metrics)):
            self.metrics[i].on_batch_end(output, target)

    def _metrics_epoch_end(self):
        for i in range(len(self.metrics)):
            self.metrics[i].on_epoch_end()

    def _log_history(self, phase, loss):
        self.history[phase]["loss"] = self.history[phase].get("loss", [])
        self.history[phase]["loss"].append(loss)
        for i in range(len(self.metrics)):
            self.history[phase][self.metrics[i].name] =\
                self.history[phase].get(self.metrics[i].name, [])
            self.history[phase][self.metrics[i].name]\
                .append(self.metrics[i].value)

    def _update_tensorboard(self):
        for k in self.history["train"].keys():
            self.tb_writer.add_scalars(k, {"train":
                self.history["train"][k][-1],
                "val": self.history["val"][k][-1]},
                self.cur_epoch)

        self.tb_writer.add_scalars("loss", {"train":
            self.history["train"]["loss"][-1], "val":
            self.history["val"]["loss"][-1]}, self.cur_epoch)

        self.tb_writer.flush()

    def _train_epoch(self, data):
        self.model.train()
        epoch_loss = 0
        data_len = 0
        self._metrics_epoch_begin()
        for _, sample in enumerate(data):
            self.optimizer.zero_grad()
            output, target, loss = self.forward(sample)
            self.backward(loss)

            batch_loss = loss.detach().item()
            epoch_loss += batch_loss * output.size(0)
            data_len += output.size(0)

            self._metrics_batch_end(output, target)

        epoch_loss /= data_len
        self._metrics_epoch_end()
        self._log_history("train", epoch_loss)

    def evaluate(self, data):
        self.model.eval()
        total_loss = 0
        data_len = 0
        self._metrics_epoch_begin()

        with torch.set_grad_enabled(False):
            for _, sample in enumerate(data):
                output, target, loss = self.forward(sample)

                batch_loss = loss.detach().item()
                total_loss += batch_loss * output.size(0)
                data_len += output.size(0)
                self._metrics_batch_end(output, target)

            total_loss /= data_len

        self._metrics_epoch_end()

        return total_loss, {self.metrics[i].name:
            self.metrics[i].value for i in range(len(self.metrics))}

    def _eval_epoch(self, data):
        epoch_loss, _ = self.evaluate(data)
        self._log_history("val", epoch_loss)

    def train(self, data_train, epochs, data_val=None, scheduler=None):
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.tb_writer = SummaryWriter(self.log_path)
        if not os.path.exists(self.cfg_path):
            os.mkdir(self.cfg_path)
        shutil.copy(self.cfg.__file__, self.cfg_path)

        self.total_epochs = epochs
        self.cur_epoch = 0
        if scheduler is not None:
            self.scheduler = scheduler

        progress = tqdm(total=self.total_epochs)
        for i in range(self.total_epochs):
            progress.set_description("Epoch {}/{}".format(i + 1,
                self.total_epochs))
            self._train_epoch(data_train)

            if data_val is not None:
                self._eval_epoch(data_val)

            self._update_tensorboard()
            progress.update(1)
            self.cur_epoch += 1

    def get_history(self):
        return self.history

    def predict(self, data):
        self.model.eval()
        y_pred = []
        y_prob = []

        with torch.set_grad_enabled(False):
            for _, sample in tqdm(enumerate(data), total=len(data)):
                output, _, _ = self.forward(sample)
                y_prob.append(output.detach().cpu().numpy())
                y_pred.append(np.argmax(output.detach().cpu().numpy(),
                    axis=-1))

        y_pred = np.concatenate(y_pred, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)

        return y_pred, y_prob
