import torch
from torchvision import datasets, transforms
import numpy as np
import cv2

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
    num_epochs = 25

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
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


if __name__ == '__main__':
    # Define the transformation to convert images to PyTorch tensors
    transform = transforms.Compose([transforms.ToTensor()])
    # Load the MNIST dataset with the specified transformation
    mnist_pytorch = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    SIZE = 100
    indices = list(range(SIZE))
    mnist_subset = torch.utils.data.Subset(mnist_pytorch, indices)

    # Create a DataLoader to load the dataset in batches
    train_loader_pytorch = torch.utils.data.DataLoader(mnist_subset, batch_size=SIZE, shuffle=False)

    for batch in train_loader_pytorch:
        (images, labels) = batch


    net = LeNet()

    train_neural_net(net, images, labels)

    mnist_test_subset = torch.utils.data.Subset(mnist_pytorch, list(range(100)))
    test_loader = torch.utils.data.DataLoader(mnist_test_subset, batch_size=100, shuffle=False)

    total = 0
    correct = 0

    net.eval()
    with torch.no_grad():
        for test_images, labels in test_loader:
            for i in range(len(test_images)):  # Loop through each image in the batch
                test_image = test_images[i]  # Get one image
                label = labels[i]  # Get corresponding label

                # Forward pass
                output_layer = net(test_image)
                output_softmax = torch.nn.functional.softmax(output_layer, dim=0)
                predicted_label = torch.argmax(output_softmax, dim=0)

                total += 1
                if predicted_label == label:
                    correct += 1

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')




    #webcam()


def image_to_tensor(image):
    transform = transforms.Compose([transforms.ToTensor()])

    test_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.GaussianBlur(test_image, (5, 5), 0)
    ret, test_image = cv2.threshold(test_image, 120, 255, cv2.THRESH_TOZERO)  # thresholding
    test_image = cv2.resize(test_image, (28, 28))  # Gray scale and 28 op 28
    cv2.imshow('test image', test_image)
    cv2.waitKey(0)

    test_image_tensor = transform(test_image)
    return test_image_tensor
def webcam():
    while True:
        cam = cv2.VideoCapture(0)
        result, image = cam.read()
        if result:
            test_image_tensor = image_to_tensor(image)
            print(test_image_tensor)
            with torch.no_grad():
                output_layer = net(test_image_tensor)
                output_softmax = torch.nn.functional.softmax(output_layer, dim=0)
                predicted_label = np.argmax(output_softmax)
                print(f"LeNet : {predicted_label}")
                #cv2.imshow('handwritten digit', cv2.resize(image, (400, 400)))
                #cv2.waitKey(0)
        else:
            print("Webcam failed")