import os

import numpy as np


def write_metrics_to_txt(metrics, path):
    with open(os.path.join(path, "metrics.txt"), "w") as f:
        for k, v in metrics.items():
            print("{}: {:.4f}".format(k, v))
            f.writelines("{}: {:.4f}\n".format(k, v))


class BaseMetric(object):

    def __init__(self, mode, num_classes):
        self.mode = mode
        self.num_classes = num_classes
        self.name = self.__class__.__name__
        self.val = 0

    def calc_metric(self):
        raise NotImplementedError("Method calc_metric not implemented.")

    def to_categorical(self, target):
        return np.eye(self.num_classes)[target]
    
    def on_epoch_begin(self):
        self.output = []
        self.target = []

    def on_batch_end(self, output, target):
        self.output.append(output.detach().cpu().numpy())
        self.target.append(target.detach().cpu().numpy())
    
    def process_output_target(self):
        pass
    
    def on_epoch_end(self):
        self.output = np.concatenate(self.output, axis=0)
        self.target = np.concatenate(self.target, axis=0)
        self.process_output_target()
        if len(self.target) == 0:
            self.value = 0
        else:
            try:
                self.value = self.calc_metric()
            except ValueError:
                self.value = 0
        return self.value
