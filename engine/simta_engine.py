from engine.base_engine import BaseEngine
from utils.cuda import put_var_on_gpu


class SimTAEngine(BaseEngine):

    def forward(self, sample):
        t, x, y, y_true = sample
        t = put_var_on_gpu(t, self.cfg.devices)
        x = put_var_on_gpu(x, self.cfg.devices)
        y = put_var_on_gpu(y, self.cfg.devices)
        y_true = put_var_on_gpu(y_true, self.cfg.devices)
        output = self.model(x, t)
        loss = self.loss_fn(output, y)

        return output, y_true, loss
