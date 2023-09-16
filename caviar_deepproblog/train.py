import torch
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_fact_accuracy
from caviar_deepproblog.markov_models.graph_semiring import BoundEntropySemiring
from caviar_deepproblog.markov_models.markov_model import MarkovModel
from caviar_deepproblog.data.caviar_utils import load_fold
from caviar_deepproblog.neural.caviar_net import CaviarNet
from caviar_deepproblog.data.caviar import CaviarVideos, CaviarDataset

import numpy as np
import random

if not torch.cuda.is_available():
    torch.set_num_threads(1)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

fold_data = load_fold(1)

# caviar_net = CaviarNet(4, 5, 256, 2)
caviar_net = CaviarNet(4, 5, 128, 2)
network = Network(caviar_net, "caviar_net")
network.optimizer = torch.optim.Adam(caviar_net.parameters(), lr=1e-5)

model = MarkovModel(
    # "/home/whatever/programms/caviar-deepproblog/caviar_deepproblog/problog_files/caviar.pl",
    "/home/nkatz/dev/caviar-deepproblog/caviar_deepproblog/problog_files/caviar.pl",
    [network],
)
model.set_engine(ExactEngine(model), semiring=BoundEntropySemiring)
model.add_tensor_source("train", CaviarVideos(fold_data["train"]["videos"]))
model.add_tensor_source("test", CaviarVideos(fold_data["test"]["videos"]))

train_dataset = CaviarDataset(fold_data["train"]["labels"], complex_event="meeting")
test_dataset = CaviarDataset(fold_data["test"]["labels"], complex_event="meeting")
loader = DataLoader(train_dataset, 1, False)


train_model(model, loader, 50, loss_function_name="cross_entropy")
model.save_state("snapshot/model.sve")

print(get_fact_accuracy(model, test_dataset))
