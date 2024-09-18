import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Define the function we want to approximate (sine in this case)
def true_function(x):
    return np.sin(x)


# Define the neural network
class FunctionApproximator(nn.Module):
    def __init__(self):
        super(FunctionApproximator, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(64, 64)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(64, 1)  # Second hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU activation for hidden layers
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for the output
        return x


# Hyperparameters
learning_rate = 0.001
num_epochs = 1000

# Create the model, loss function, and optimizer
model = FunctionApproximator()
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Generate training data
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # Input data
y_train = true_function(x_train)  # True output data

# Convert training data to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Step 4: Train the Neural Network
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss for every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 5: Testing the Model
model.eval()  # Switch to evaluation mode
with torch.no_grad():
    predicted = model(x_train_tensor).numpy()

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_train, y_train, label='Sine function', color='b')
plt.plot(x_train, predicted, label='Neural Network Approximation', color='r')
plt.title('Sine function approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
