import torch
import torch.nn as nn
from problog.logic import Constant, Term


class CaviarNet(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, video_tensor: torch.Tensor, personID: Term, timestep: Constant):
        # the aim of each LSTM forward pass is to perform simple event recognition
        # for a single person. Since we are interested in queries of the form "are
        # the two people present in timestep 5 of video 001 interacting?", each query
        # requires us to perform simple event recognition twice, once for each of
        # the two people present. Therefore, depending on the personID passed (p1 or
        # p2), we use the respective part of the input video tensor.

        if personID.functor == "p1":
            lstm_input = video_tensor[:, :5]
        elif personID.functor == "p2":
            # For the second person ignore the last feature since it is the distance.
            # We use it elsewhere no need for the NN to take it into account.
            lstm_input = video_tensor[:, 5:-1]
        else:
            raise ValueError("Parameter 'personID' should be either p1 or p2")

        # generate output (simple event) for the given person by passing the input
        # first through an LSTM network with two LSTMs and then through an MLP
        # network with two fully-connected layers wtih a ReLU and Softmax activation
        lstm_output, _ = self.lstm(lstm_input)
        output = self.mlp(lstm_output)

        # we're currently doing the very dumb thing of having individual queries for
        # each timestep in the sequence (i.e. we evaluate the LSTM 50 times for the
        # same 50x10 tensor because we don't yet know how to perform 50 queries
        # within the same LSTM evaluation). For this reason, each forward pass returns
        # only a single timestep from the entire generated output. Wasteful and sad.
        return output[int(timestep)]


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
            mlp_input = frame[5:-1]
        else:
            raise ValueError("Parameter 'personID' should be either p1 or p2")

        x = self.relu(self.fc1(mlp_input))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))

        return x
