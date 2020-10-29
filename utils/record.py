import os
import pickle


def pickle_history(history, path):
    train_loss = history["train"]["loss"]
    val_loss = history["val"]["loss"]

    pickle.dump(train_loss, open(os.path.join(path,
        "train_loss.pickle"), "wb"))
    pickle.dump(val_loss, open(os.path.join(path, "val_loss.pickle"), "wb"))
