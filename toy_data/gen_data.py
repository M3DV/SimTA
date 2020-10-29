import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from toy_data.toy_data import ToyData


def _generate_rnd_sin_param(cfg):
    w = np.random.randint(cfg.w_min, cfg.w_max, cfg.num_freq)
    b = np.random.uniform(cfg.b_min, cfg.b_max, cfg.num_freq)
    alpha = np.random.uniform(cfg.alpha_min, cfg.alpha_max, cfg.num_freq)
    beta = np.random.uniform(cfg.beta_min, cfg.beta_max, cfg.num_freq)

    return w, b, alpha, beta


def generate_toy_data(cfg):
    w, b, alpha, beta = _generate_rnd_sin_param(cfg)
    toy_data = ToyData(w, b, alpha, beta, cfg.eps)
    params = (w, b, alpha, beta, cfg.eps)
    pickle.dump(params, open(os.path.join(cfg.save_path,
        "params.pickle"), "wb"))

    t = np.zeros((cfg.num_total_samples, cfg.num_pt_per_sample),
        dtype=np.float32)
    x = np.zeros_like(t, dtype=np.float32)
    y = np.zeros((cfg.num_total_samples, cfg.num_test_pt), dtype=np.float32)
    y_true = np.zeros_like(y, dtype=np.float32)

    for i in range(cfg.num_total_samples):
        t[i, 1:] = np.random.uniform(0, cfg.duration_per_step,
            cfg.num_pt_per_sample - 1)
        t_series = np.cumsum(t[i, :])
        x[i, :] = toy_data.sample(t_series)
        y[i, :] = toy_data.sample(np.linspace(t_series[-1] + 1,
            t_series[-1] + cfg.num_test_pt, cfg.num_test_pt))
        y_true[i, :] = toy_data.generate_denoised_series(
            np.linspace(t_series[-1] + 1, t_series[-1] + cfg.num_test_pt,
            cfg.num_test_pt))
    
    idx_train, idx_val = train_test_split(np.arange(cfg.num_total_samples),
        test_size=cfg.val_pct)

    pickle.dump(t, open(os.path.join(cfg.save_path, "t.pickle"), "wb"))
    pickle.dump(x, open(os.path.join(cfg.save_path, "x.pickle"), "wb"))
    pickle.dump(y, open(os.path.join(cfg.save_path, "y.pickle"), "wb"))
    pickle.dump(y_true, open(os.path.join(cfg.save_path, "y_true.pickle"),
        "wb"))
    pickle.dump(idx_train, open(os.path.join(cfg.save_path,
        "idx_train.pickle"), "wb"))
    pickle.dump(idx_val, open(os.path.join(cfg.save_path,
        "idx_val.pickle"), "wb"))


if __name__ == "__main__":
    import toy_data.toy_data_cfg as cfg

    generate_toy_data(cfg)
