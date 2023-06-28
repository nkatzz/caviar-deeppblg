import re
import torch
import numpy as np
from deepproblog.network import Network
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from sklearn.metrics import classification_report
from markov_models.markov_model import MarkovModel
from data.caviar_utils import load_fold
from neural.caviar_net import CaviarNet
from data.caviar import CaviarVideos, CaviarDataset

import numpy as np
import random
import os
import sys

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

lr = 0.0001
epochs = 2
if len(sys.argv) == 2:
    model_name = sys.argv[1]

def custom_cm(subset):
    if subset == 'train':
        dataset = train_dataset
    elif subset == 'test':
        dataset = test_dataset
    else:
        pass
        #handle exception
    
    j = 0
    i = len(dataset)
    print('Loading labels...')
    actual = []
    for label in dataset.complex_event_labels:
        actual += label.tolist()
    actual = actual[j:i]
    print('Labels loaded')

    print('Loading queries...')
    q_list = []
    for q in range(j,i):
        q_list.append(dataset.to_many_queries(q))
    print('Queries loaded')

    print('Solving queries...')
    predicted = []
    print(f'Queries to solve {len(q_list)}')
    for batch in q_list:
        temp = [float(re.findall(r"(\d+\.\d+)", str(x))[0]) for x in model.solve(batch)]
        predicted.append(np.argmax(temp))

    print('Predictions extracted')
    print(len(predicted))
    print(len(actual))

    print(classification_report(actual,predicted))
        

problog_path = os.getcwd() + "/problog_files/caviar.pl"
complex_event_mapping = {"nointeraction": 0, "meeting": 1, "moving": 2}
inverse_complex_event_mapping = {0: "nointeraction", 1: "meeting", 2: "moving"}

for fold in range(1,4):
    fold_data = load_fold(fold)
    print(f'Testing fold {fold}')
    caviar_net = CaviarNet()
    network = Network(caviar_net, "caviar_net")
    network.optimizer = torch.optim.Adam(caviar_net.parameters(), lr=0.001)

    model = MarkovModel(
        problog_path,
        [network],
    )
    model.set_engine(ExactEngine(model))
    model.add_tensor_source("train", CaviarVideos(fold_data["train"]["videos"]))
    model.add_tensor_source("test", CaviarVideos(fold_data["test"]["videos"]))
    #model.load_state('snapshot/model_lr{lr}_epochs{epochs}.sve')
    model.load_state('snapshot/' + model_name)

    train_dataset = CaviarDataset(fold_data["train"]["labels"], complex_event="meeting")
    test_dataset = CaviarDataset(fold_data["test"]["labels"], complex_event="meeting", is_train=False)
    loader = DataLoader(train_dataset, 1, False)

    custom_cm('test')
