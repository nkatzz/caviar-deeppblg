import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from caviar_deepproblog.data.caviar_utils import load_fold


class CaviarMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, video_tensor, personID, timestep):
        frame = video_tensor[int(timestep), :]

        if personID.functor == "p1":
            mlp_input = frame[:5]
        elif personID.functor == "p2":
            mlp_input = frame[5:-2]
        else:
            raise ValueError("Parameter 'personID' should be either p1 or p2")

        x = self.relu(self.fc1(mlp_input))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return x


class CAVIARTimeSeries(Dataset):
    def __init__(self, fold_data, subset):
        self.sequences = fold_data[subset]["videos"]
        self.labels = fold_data[subset]["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


torch.manual_seed(42)
fold_data = load_fold(1)

train_dataset = CAVIARTimeSeries(fold_data, "train")
test_dataset = CAVIARTimeSeries(fold_data, "test")

train_dl = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dl = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)
# X_train, y_train = fold_data["train"]["videos"], fold_data["train"]["labels"]
# train_dl2 = DataLoader(list(zip(X_train, y_train)), batch_size=32, shuffle=True)


caviar_MLP = CaviarMLP(input_size=5, num_classes=4)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(caviar_MLP.parameters(), lr=1e-3)
num_epochs = 20


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
        average_test_loss = 0
        for test_inputs, test_labels in test_dl:
            test_outputs = caviar_MLP(test_inputs)
            test_loss = loss_fn(test_outputs, test_labels)
            average_test_loss += test_loss

        print(f"Average test loss: {average_test_loss / len(test_dl)}")
