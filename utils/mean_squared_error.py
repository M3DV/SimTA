from sklearn.metrics import mean_squared_error

from utils.base_metric import BaseMetric


class MeanSquaredError(BaseMetric):

    def __init__(self):
        self.name = self.__class__.__name__

    def calc_metric(self):
        return mean_squared_error(self.target, self.output)
