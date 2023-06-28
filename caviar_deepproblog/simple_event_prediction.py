import cv2
import torch
import numpy as np
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import torchmetrics
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optimizers
from torch.utils.data import Dataset, DataLoader, random_split
from caviar_deepproblog.data.caviar_vision_data import CaviarVisionDataset
from caviar_deepproblog.neural.caviar_net import SupervisedCaviarCNN
from rich.progress import track


class SupervisedCaviarVision(Dataset):
    def __init__(self, window_size, window_stride, desired_image_size):
        vision_dataset = CaviarVisionDataset(
            window_size=window_size,
            window_stride=window_stride,
            preprocess=True,
            desired_image_size=desired_image_size,
        )

        self.simple_event_labels = torch.Tensor(
            self.flatten(vision_dataset.simple_event_labels)
        ).type(torch.LongTensor)

        self.input_images = torch.stack(self.flatten(vision_dataset.input_images))

    def flatten(self, nested_list):
        # here nested lists have the following shape:
        # (num_sequences x num_timesteps x 2 people x obj)
        # num_timesteps = window_size
        # for input_images obj is a 3D numpy array representing a cropped bounding box
        # for simple_events obj is an int representing an SE

        flat_list = []
        for sequence in nested_list:
            for timestep in sequence:
                for obj in timestep:
                    flat_list.append(obj)

        return flat_list

    def show_training_example(self, example_indices):
        inverse_SE_mapping = {
            0: "active",
            1: "inactive",
            2: "walking",
            3: "running",
        }

        fig, ax = plt.subplots(1, len(example_indices))

        for i, idx in enumerate(example_indices):
            image = self.input_images[idx]
            label = self.simple_event_labels[idx]

            ax[i].set_title(inverse_SE_mapping[label.item()])
            ax[i].imshow(torch.permute(image, (1, 2, 0)))

        plt.show()

    def __len__(self):
        return len(self.simple_event_labels)

    def __getitem__(self, idx):
        se_label = self.simple_event_labels[idx]
        image = self.input_images[idx]

        return image, se_label


if __name__ == "__main__":
    f1_score = torchmetrics.F1Score(task="multiclass", num_classes=4)

    dataset = SupervisedCaviarVision(
        window_size=24,
        window_stride=24,
        desired_image_size=(80, 80),
    )

    train_data, test_data = random_split(dataset, [10620, 3540])
    train_dl = DataLoader(train_data, batch_size=256, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=256, shuffle=True)

    cnn = SupervisedCaviarCNN(num_classes=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizers.Adam(cnn.parameters(), lr=1e-3)
    num_epochs = 20

    for epoch in track(range(num_epochs)):
        print(f"Epoch {epoch + 1}/{num_epochs} \t --- ", end="")
        epoch_loss = 0
        cnn.train()
        for batch_idx, (inputs, labels) in enumerate(train_dl):
            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f"\t train loss: {round(epoch_loss, 4)}")

    cnn.eval()
    for test_inputs, test_labels in test_dl:
        test_outputs = cnn(test_inputs)
        f1_metrics = f1_score(test_outputs, test_labels)

    f1_metrics = f1_score.compute()
    print(
        f"Test F1: {round(f1_metrics.item(), 4)}",
    )
