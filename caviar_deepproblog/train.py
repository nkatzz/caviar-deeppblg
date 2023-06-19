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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
problog_path = "/home/blackbeard/Documents/caviar-deepproblog/caviar_deepproblog/problog_files/caviar.pl"

fold_data = load_fold(1)

caviar_net = CaviarNet(4, 5, 128, 1)
network = Network(caviar_net, "caviar_net")
network.optimizer = torch.optim.Adam(caviar_net.parameters(), lr=0.001)

model = MarkovModel(
    problog_path,
    [network],
)
model.set_engine(ExactEngine(model))
model.add_tensor_source("train", CaviarVideos(fold_data["train"]["videos"]))
model.add_tensor_source("test", CaviarVideos(fold_data["test"]["videos"]))

train_dataset = CaviarDataset(fold_data["train"]["labels"], complex_event="meeting")
test_dataset = CaviarDataset(fold_data["test"]["labels"], complex_event="meeting")
loader = DataLoader(train_dataset, 1, False)

train_model(model, loader, 5, loss_function_name="cross_entropy")
model.save_state('snapshot/model.sve')

print(get_fact_accuracy(model, test_dataset))
