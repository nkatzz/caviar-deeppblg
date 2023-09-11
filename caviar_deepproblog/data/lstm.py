import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.metrics import classification_report, confusion_matrix

# Load the data
with open('caviar_folds.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract the first fold's train and test data
fold1 = data['fold1']
train_data = fold1['train']
test_data = fold1['test']

# Prepare training and testing tensors
X_train = torch.cat([item['concat_tensor'] for item in train_data], dim=0)
y_train = torch.cat([item['complex_labels'] for item in train_data], dim=0)
X_test = torch.cat([item['concat_tensor'] for item in test_data], dim=0)
y_test = torch.cat([item['complex_labels'] for item in test_data], dim=0)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Hyperparameters
input_dim = X_train.size(2)
hidden_dim = 256
num_epochs = 20000
learning_rate = 0.001

# Initialize the model, loss, and optimizer
model = LSTMModel(input_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    outputs = model(X_train)
    outputs = outputs.view(-1, 3)  # Reshape to 2D tensor
    targets = y_train.view(-1)     # Flatten targets
    loss = criterion(outputs, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Model evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_outputs = test_outputs.view(-1, 3)
    test_targets = y_test.view(-1)
    _, predicted = torch.max(test_outputs, 1)
    report = classification_report(test_targets, predicted)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", confusion_matrix(test_targets, predicted))

