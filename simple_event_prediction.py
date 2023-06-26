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
from caviar_deepproblog.data.caviar_vision import (
    CaviarVisionDataset,
    inverse_simple_event_mapping,
)
from caviar_deepproblog.neural.caviar_net import CaviarCNN


class SupervisedCaviarVision(Dataset):
    def __init__(self, dataset_root, window_size, window_stride, desired_image_size):
        vision_dataset = CaviarVisionDataset(
            dataset_root=dataset_root,
            window_size=window_size,
            window_stride=window_stride,
        )

        self.simple_event_labels = torch.Tensor(
            self.flatten(vision_dataset.simple_event_labels)
        ).type(torch.LongTensor)

        self.input_images = torch.stack(
            [
                self.preprocess_image(image, desired_image_size)
                for image in self.flatten(vision_dataset.input_images)
            ]
        )

        print()

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

    def preprocess_image(self, image, desired_image_size):
        resized = cv2.resize(image, dsize=desired_image_size)
        reshaped = np.transpose(resized, (2, 0, 1))

        return torch.Tensor(reshaped).to(torch.float)

    def show_training_example(self, example_indices):
        fig, ax = plt.subplots(1, len(example_indices))

        for i, idx in enumerate(example_indices):
            image = self.input_images[idx]
            label = self.simple_event_labels[idx]

            # torch uses (num_channels, height, width) so reshape to
            # (height, width, num_channels), also cast to int for showing
            image = torch.permute(image, (1, 2, 0)).to(torch.uint8)

            # get the simple event label from its numeric value to set as title
            label = inverse_simple_event_mapping[label.item()]

            ax[i].set_title(label)
            ax[i].imshow(image)

        plt.show()

    def __len__(self):
        return len(self.simple_event_labels)

    def __getitem__(self, idx):
        se_label = self.simple_event_labels[idx]
        image = self.input_images[idx]

        return image, se_label


if __name__ == "__main__":
    caviar_root = "/home/yuzer/.cache/cached_path/3d7268fd95461fe356087696890c33afe4a1257e48773d5e3cc6e06d1f505a55.4baaf2515ddb1b1533af48a43c660a60fa029edfc3562069cb4afcbcdb9081e8-extracted/caviar_videos"

    dataset = SupervisedCaviarVision(
        dataset_root=caviar_root,
        window_size=24,
        window_stride=24,
        desired_image_size=(85, 85),
    )

    # f1_torcheval = MulticlassF1Score(num_classes=4)
    f1_torchmetrics = torchmetrics.F1Score(task="multiclass", num_classes=4)

    train_data, test_data = random_split(dataset, [0.8, 0.2])

    train_dl = DataLoader(train_data, batch_size=256, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=256, shuffle=True)

    cnn = CaviarCNN(num_classes=4)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optimizers.Adam(cnn.parameters(), lr=1e-3)
    num_epochs = 20

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} --- ", end="")
        epoch_loss = 0
        cnn.train()
        for batch_idx, (inputs, labels) in enumerate(train_dl):
            outputs = cnn(inputs)
            loss = loss_fn(outputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"done \t train loss: {round(epoch_loss, 3)}")

    cnn.eval()
    for test_inputs, test_labels in test_dl:
        test_outputs = cnn(test_inputs)
        f1_metrics = f1_torchmetrics(test_outputs, test_labels)

    f1_metrics = f1_torchmetrics.compute()
    print(
        f"\ttest F1 (torchmetrics): {round(f1_metrics.item(), 3)}",
    )
