import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optimizers
import torchvision.transforms as T
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
from caviar_deepproblog.neural.caviar_net import CaviarCNN
from caviar_deepproblog.data.caviar_vision import (
    CaviarVisionDataset,
    inverse_simple_event_mapping,
)
from torcheval.metrics import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)


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
        # opencv uses the BGR convention whereas PIL uses the RGB convention
        # so have to convert to RGB before converting to PIL image
        image_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        transforms = T.Compose(
            [
                T.Resize(size=desired_image_size),
                T.PILToTensor(),
            ]
        )

        resized = transforms(image_PIL).float()

        return resized

    def show_training_example(self, idx):
        image = self.input_images[idx]
        label = self.simple_event_labels[idx]

        image = T.ToPILImage()(image)

        label = inverse_simple_event_mapping[label.item()]

        image.show(label)

    def __len__(self):
        return len(self.simple_event_labels)

    def __getitem__(self, idx):
        se_label = self.simple_event_labels[idx]
        image = self.input_images[idx]

        return image, se_label


if __name__ == "__main__":
    caviar_root = "/home/yuzer/.cache/cached_path/3d7268fd95461fe356087696890c33afe4a1257e48773d5e3cc6e06d1f505a55.4baaf2515ddb1b1533af48a43c660a60fa029edfc3562069cb4afcbcdb9081e8-extracted/caviar_videos"

    # caviar_root = os.path.join(
    #     cached_path.cached_path(
    #         "https://users.iit.demokritos.gr/~nkatz/caviar_videos.zip",
    #         extract_archive=True,
    #     ),
    #     "caviar_videos",window_size
    # )

    dataset = SupervisedCaviarVision(
        dataset_root=caviar_root,
        window_size=24,
        window_stride=24,
        desired_image_size=(80, 80),
    )
    test_metrics = [
        MulticlassAccuracy(),
        MulticlassPrecision(),
        MulticlassRecall(),
        MulticlassF1Score(),
    ]

    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(k_fold.split(np.arange(len(dataset)))):
        print("Fold {}".format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_dl = DataLoader(dataset, batch_size=256, sampler=train_sampler)
        test_dl = DataLoader(dataset, batch_size=256, sampler=test_sampler)

        cnn = CaviarCNN(num_classes=4)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optimizers.Adam(cnn.parameters(), lr=1e-3)

        for epoch in range(10):
            print(f"Epoch {epoch + 1}/10 --- ", end="")
            cnn.train()
            for batch_idx, (inputs, labels) in enumerate(train_dl):
                outputs = cnn(inputs)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # # compute and print training metrics
            #     outputs_int = []

            #     # update all metrics for the new batch
            #     for output in outputs:
            #         outputs_int.append(torch.argmax(output).float())

            #     [
            #         metric.update(torch.Tensor(outputs_int), labels)
            #         for metric in train_metrics
            #     ]

            #     # report performance every 10 batches
            #     if (batch_idx + 1) % 5 == 0:
            #         print(
            #             "Epoch {}/{}, Batch {}/{} --- acc: {:.4f}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(
            #                 epoch + 1,
            #                 10,
            #                 batch_idx + 1,
            #                 len(train_dl),
            #                 *[metric.compute() for metric in train_metrics],
            #             )
            #         )
            # # reset all metrics before the next epoch
            # [metric.reset() for metric in train_metrics]

            with torch.no_grad():
                cnn.eval()
                for test_inputs, test_labels in test_dl:
                    test_outputs = cnn(test_inputs)
                    test_outputs_int = []

                    for output in test_outputs:
                        test_outputs_int.append(torch.argmax(output).float())

                    [
                        metric.update(torch.Tensor(test_outputs_int), test_labels)
                        for metric in test_metrics
                    ]

                epoch_performance = [metric.compute().item() for metric in test_metrics]
                epoch_performance = [round(metric, 3) for metric in epoch_performance]

            print(
                "acc: {:.4f}, pre: {:.4f}, rec: {:.4f}, f1: {:.4f}".format(
                    *epoch_performance
                )
            )
