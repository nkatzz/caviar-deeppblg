import torch
from deepproblog.network import Network
from deepproblog.train import train_model
from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine
from deepproblog.evaluate import get_fact_accuracy
from caviar_deepproblog.markov_models.markov_model import MarkovModel
from caviar_deepproblog.neural.caviar_net import CaviarCNN
from caviar_deepproblog.data.caviar_vision_data import CaviarVisionDataset
from caviar_deepproblog.data.dpl_caviar_interface import (
    CaviarImagePairs,
    CaviarVisionDPLDataset,
)
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

dataset = CaviarVisionDataset(
    window_size=24,
    window_stride=24,
    preprocess=True,
    desired_image_size=(80, 80),
)

(
    input_images_train,
    input_images_test,
    CE_labels_train,
    CE_labels_test,
) = train_test_split(
    dataset.input_images,
    dataset.complex_event_labels,
    train_size=0.75,
)

caviar_cnn = CaviarCNN(num_classes=4)
network = Network(caviar_cnn, "caviar_cnn")
network.optimizer = torch.optim.Adam(caviar_cnn.parameters(), lr=0.001)

model = MarkovModel(
    os.path.join(os.getcwd(), "caviar_deepproblog/problog_files/caviar.pl"),
    [network],
)

model.set_engine(ExactEngine(model))
model.add_tensor_source("train", CaviarImagePairs(input_images_train))
model.add_tensor_source("test", CaviarImagePairs(input_images_test))

train_dataset = CaviarVisionDPLDataset(
    subset="train",
    complex_event_labels=CE_labels_train,
    complex_event="meeting",
)
test_dataset = CaviarVisionDPLDataset(
    subset="test",
    complex_event_labels=CE_labels_test,
    complex_event="meeting",
)


loader = DataLoader(train_dataset, 1, False)


# ------------------------------- THIS WILL CRASH -------------------------------
# need to figure out how to pass distance between persons along with their images
# currently assume it should be computed in the init of CaviarVisionDataset and
# then passed along with the image tensor as part of one term? maybe this is handled
# in the to_query or in the tensor_source __getitem__

train_model(model, loader, 5, loss_function_name="cross_entropy")

print(get_fact_accuracy(model, test_dataset))
