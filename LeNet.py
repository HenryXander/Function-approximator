import torch
from torchvision import datasets, transforms
import numpy as np

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2) #1 channel because black and white image (rgb image would need 3 channels)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.fc1 = torch.nn.Linear(5*5*16, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, input):
        c1 = torch.nn.functional.sigmoid(self.conv1(input))
        p2 = torch.nn.functional.avg_pool2d(c1, (2, 2), stride=2)
        c3 = torch.nn.functional.sigmoid(self.conv2(p2))
        p4 = torch.nn.functional.avg_pool2d(c3, (2, 2), stride=2)
        p4 = torch.flatten(p4)
        f5 = torch.nn.functional.sigmoid(self.fc1(p4))
        f6 = torch.nn.functional.sigmoid(self.fc2(f5))
        output = self.fc3(f6)
        return output




def train_neural_net(NN, x_train_tensor, y_train_tensor):
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 200

    # Create the model, loss function, and optimizer
    model = NN
    criterion = torch.nn.functional.cross_entropy  # cross entropy voor classificatie
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    x_train_tensor = x_train_tensor.float()
    y_train_tensor = y_train_tensor.long() #long for cross entropy


    #Train the Neural Network
    for epoch in range(num_epochs):
        y_train_label = iter(y_train_tensor)
        for x_train_img in x_train_tensor:
            # Forward pass
            outputs = model(x_train_img)
            loss = criterion(outputs, next(y_train_label))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print the loss for every 100 epochs
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')



if __name__ == '__main__':
    # Define the transformation to convert images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])
    # Load the MNIST dataset with the specified transformation
    mnist_pytorch = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    SIZE = 50
    indices = list(range(SIZE))
    mnist_subset = torch.utils.data.Subset(mnist_pytorch, indices)

    # Create a DataLoader to load the dataset in batches
    train_loader_pytorch = torch.utils.data.DataLoader(mnist_subset, batch_size=SIZE, shuffle=False)

    for batch in train_loader_pytorch:
        (images, labels) = batch


    net = LeNet()

    train_neural_net(net, images, labels)

    test_subset = torch.utils.data.Subset(mnist_pytorch, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=10, shuffle=False)

    for batch in test_dataloader:
        (test_images, test_labels) = batch

        net.eval()
        with torch.no_grad():
            test_label = iter(test_labels)
            for test_image in test_images:
                output_layer = net(test_image)
                output_softmax = torch.nn.functional.softmax(output_layer, dim=0)
                predicted_label = np.argmax(output_softmax)
                print(f"\nLeNet : {predicted_label}\nlabel : {next(test_label)}")