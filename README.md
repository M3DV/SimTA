# **Sim**ple **T**emporal **A**ttention (SimTA)

We propose Simple Temporal Attention (SimTA) module to process the asynchronous time series. Experiments on synthetic dataset validate the superiory of SimTA over standard RNN-based approaches. Furthermore, we experiment the proposed method on an in-house, retrospective dataset of real-world non-small cell lung cancer patients under anti-PD-1 immunotherapy. The proposed method achieves promising performance on predicting the immunotherapy response. Notably, our predictive model could further stratify low-risk and high-risk patients in terms of long-term survival.

![SimTA Illustration](etc/simta_illustration.png)

For more details, please refer to our paper: [MIA-Prognosis: A Deep Learning Framework to Predict Therapy Response](https://arxiv.org/pdf/2010.04062.pdf).

## Code Structure
* [`SimTA/`](./)
    * [`config/`](config/): Training configurations
    * [`dataset/`](dataset/): Toy data generator, PyTorch dataset and toy data pickles used in experiments.
    * [`engine/`](engine/): Keras-like training and evaluation API
    * [`etc/`](etc/): images for README.md
    * [`models/`](models/): PyTorch Implementation of SimTA modules and LSTM model as comparison.
    * [`run/`](/run): Training and evaluation scripts
    * [`utils/`](/utils): Utility functions

## Requirements
```
pytorch>=1.3.0
numpy>=1.18.0
scikit-learn>=0.23.2
tqdm>=4.38.0
```

## Usage
To train the SimTA model on the toy dataset, run:
```bash
python -m run.main_simta --logdir <logging_directory>
```
Tensorboard logs, configuration files and trained PyTorch models are saved under the `logging_directory` that you specify.
To train LSTM-based approaches for comparison, run:
```bash
python -m run.main_lstm --logdir <logging_directory>
python -m run.main_lstm_time_interval --logdir <logging_directory>
python -m run.main_lstm_time_stamp --logdir <logging_directory>
```

# TODO
- Evaluation script;
- Tensorboard screenshot.