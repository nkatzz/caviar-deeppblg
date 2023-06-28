import markov_models.patched_graph_semiring
import sys

sys.modules['deepproblog.semiring.graph_semiring'] = markov_models.patched_graph_semiring

import torch
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_fact_accuracy
from markov_models.markov_model import MarkovModel
from data.caviar_utils import load_fold
from neural.caviar_net import CaviarNet
from data.caviar import CaviarVideos, CaviarDataset

import numpy as np
import random
import os

if len(sys.argv) == 3:
    lr = float(sys.argv[1])
    epochs = int(sys.argv[2])
elif len(sys.argv) == 4:
    lr = float(sys.argv[1])
    epochs = int(sys.argv[2])
    batch_size = int(sys.argv[3])
else:
    lr = 0.00001
    epochs = 70
    batch_size = 1

print(f'lr: {lr}')
print(f'epochs: {epochs}')
print(f'batch size: {batch_size}')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
problog_path = os.getcwd() + "/problog_files/caviar.pl"

fold_data = load_fold(1)
caviar_net = CaviarNet()
network = Network(caviar_net, "caviar_net")
network.optimizer = torch.optim.Adam(caviar_net.parameters(), lr)

model = MarkovModel(
    problog_path,
    [network],
)
model.set_engine(ExactEngine(model))
model.add_tensor_source("train", CaviarVideos(fold_data["train"]["videos"]))
model.add_tensor_source("test", CaviarVideos(fold_data["test"]["videos"]))

train_dataset = CaviarDataset(fold_data["train"]["labels"], complex_event="meeting")
test_dataset = CaviarDataset(fold_data["test"]["labels"], complex_event="meeting", is_train = False)
loader = DataLoader(train_dataset, batch_size, False)
print(len(train_dataset))
train_model(model, loader, epochs, loss_function_name="cross_entropy")
model.save_state(f'snapshot/model_lr{lr}_epochs{epochs}.sve')
print(get_fact_accuracy(model, test_dataset))
