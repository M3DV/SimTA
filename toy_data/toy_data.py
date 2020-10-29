from math import pi

import numpy as np


class ToyData(object):

    def __init__(self, w, b, alpha, beta, eps=1):
        self.w = w
        self.b = b
        self.alpha = alpha
        self.beta = beta
        assert len(self.w) == len(self.b) == len(self.alpha) == len(self.beta)
        self.eps = eps
    
    def _compute_output(self, x):
        y = np.sum([self.alpha[i] * np.sin(self.w[i] * x * pi + self.b[i])
            + self.beta[i] for i in range(len(self.w))])

        return y
    
    def _compute_noisy_output(self, x):
        y = self._compute_output(x)
        noise = np.random.normal() * self.eps

        return y + noise
    
    def generate_denoised_series(self, x_series):
        y_series = np.array([self._compute_output(x_series[i])
            for i in range(len(x_series))])
        
        return y_series

    def sample(self, x_series):
        y_series = np.array([self._compute_noisy_output(x_series[i])
            for i in range(len(x_series))])
        
        return y_series


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    w_min, w_max = -5, 5
    b_min, b_max = -5, 5
    num_freq = 5
    eps = 0.1
    w = np.random.uniform(w_min, w_max, num_freq)
    b = np.random.uniform(b_min, b_max, num_freq)
    toy_data = ToyData(w, b, eps)
    x_min = 0
    x_max = 100
    len_x = 10
    x = np.random.uniform(x_min, x_max, 1000)
    y = toy_data.generate_denoised_series(x_min, x_max, 1000)
    y_noisy = toy_data.sample(x)
    plt.plot(np.linspace(x_min, x_max, 1000), y)
    plt.scatter(x, y_noisy)
    plt.show()
