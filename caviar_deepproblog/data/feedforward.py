import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, confusion_matrix

# Load data
with open('caviar_folds.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract fold1 data
fold1 = data['fold1']
train_data = fold1['train']
test_data = fold1['test']

# Data Preparation
X_train = torch.stack([item['concat_tensor'] for item in train_data]).squeeze(-1).view(-1, 24*11)  # Flattening the data
y_train = torch.stack([item['complex_labels'] for item in train_data])

X_test = torch.stack([item['concat_tensor'] for item in test_data]).squeeze(-1).view(-1, 24*11)  # Flattening the data
y_test = torch.stack([item['complex_labels'] for item in test_data])

# Neural Network Definition
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Hyperparameters
input_size = 24 * 11  # Adjusted input size
hidden_size = 50
num_classes = 3
num_epochs = 5
learning_rate = 0.001

# Neural Network and Loss
model = NeuralNet(input_size, hidden_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')

