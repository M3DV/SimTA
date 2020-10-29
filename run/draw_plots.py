import os
import pickle

import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

root_dir = "/mnt/sdb/data/simta/logs"
model_names = ["LSTM", "LSTM(i)", "LSTM(s)", "SimTA"]
history_dir = {model_names[i]: os.path.join(root_dir, model_names[i])
    for i in range(len(model_names))}

title_size = 70
legend_size = 20
tick_size = 20
label_size = 50
line_width = 5
line_style = ":"
font = fm.FontProperties(
    fname="/home/kaiming/Downloads/times-new-roman.ttf")
fig_size = (12, 10)
y_limit = 20
plt.figure(figsize=fig_size)
for k, v in history_dir.items():
    train_loss = pickle.load(open(os.path.join(v, "train_loss.pickle"), "rb"))
    plt.plot(train_loss, label=k, lw=line_width, ls=line_style)

plt.legend(fontsize=legend_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.ylim(0, y_limit)
plt.title("Training MSE", fontproperties=font, fontsize=title_size)
plt.xlabel("Epochs", fontproperties=font, fontsize=label_size)
plt.show()

plt.figure(figsize=fig_size)
for k, v in history_dir.items():
    val_loss = pickle.load(open(os.path.join(v,
        "val_loss.pickle"), "rb"))
    plt.plot(val_loss, label=k, lw=line_width, ls=line_style)

plt.legend(fontsize=legend_size)
plt.xticks(fontsize=tick_size)
plt.yticks(fontsize=tick_size)
plt.ylim(0, y_limit)
plt.title("Validation MSE", fontproperties=font, fontsize=title_size)
plt.xlabel("Epochs", fontproperties=font, fontsize=label_size)
plt.show()
