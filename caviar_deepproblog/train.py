import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from caviar_deepproblog.data.caviar_utils import load_fold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


class CaviarMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, person_features):
        x = self.relu(self.fc1(person_features))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        return x


class CAVIARTimeSeries(Dataset):
    def __init__(self, fold_data, subset):
        self.p1_features = fold_data[subset]["p1_features"]
        self.p2_features = fold_data[subset]["p2_features"]
        self.p1_simple_event_labels = fold_data[subset]["p1_simple_event_labels"]
        self.p2_simple_event_labels = fold_data[subset]["p2_simple_event_labels"]

        self.features = torch.flatten(
            torch.stack(
                [self.p1_features, self.p2_features],
            ),
            end_dim=2,
        )

        self.simple_event_labels = torch.flatten(
            torch.stack([self.p1_simple_event_labels, self.p2_simple_event_labels]),
            end_dim=2,
        )

    def __len__(self):
        return len(self.simple_event_labels)

    def __getitem__(self, idx):
        return self.features[idx], self.simple_event_labels[idx]


torch.manual_seed(42)
fold_data = load_fold(1)

train_dataset = CAVIARTimeSeries(fold_data, "train")
test_dataset = CAVIARTimeSeries(fold_data, "test")
print(
    "\n"
    f"Train inputs:\t {train_dataset.features.shape}, \ttrain outputs:\t {train_dataset.simple_event_labels.shape}\n"
    f"Test inputs:\t {test_dataset.features.shape}, \ttest outputs:\t {test_dataset.simple_event_labels.shape}"
    "\n\n"
    f"Training example:\n"
    f"\tInput features: {train_dataset.features[10]}\n"
    f"\tSimple event label: {train_dataset.simple_event_labels[10]}"
    "\n"
)

train_dl = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dl = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


caviar_MLP = CaviarMLP(input_size=5, num_classes=4)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(caviar_MLP.parameters(), lr=1e-4)
num_epochs = 150

test_losses = []
for epoch in range(num_epochs):
    caviar_MLP.train()
    print(f"Epoch {epoch + 1}/{num_epochs} \t --- ", end="")

    for inputs, labels in train_dl:
        outputs = caviar_MLP(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        caviar_MLP.eval()
        total_test_loss = 0
        for test_inputs, test_labels in test_dl:
            test_outputs = caviar_MLP(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            total_test_loss += test_loss

        average_test_loss = total_test_loss / len(test_dl)
        test_losses.append(average_test_loss)
        print(f"Average test loss: {average_test_loss}")


all_test_outputs = caviar_MLP(test_dataset.features)

print(
    classification_report(
        test_dataset.simple_event_labels.detach().numpy(),
        torch.argmax(all_test_outputs, dim=1).detach().numpy(),
        target_names=["active", "inactive", "running", "walking"],
    )
)


plt.figure()
plt.plot(test_losses)
plt.show()
