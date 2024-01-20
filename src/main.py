import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_model(train_dataset_loader):
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device to be used for training: ", device)
    
    running_loss = 0.0
    start_time = time.time()
    print('Starting model training...')
    for epoch in range(2):  # Adjust the number of epochs as needed
        for i, data in enumerate(train_dataset_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[epoch: {epoch + 1}, batch:{i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Training completed in: ', time.time() - start_time, 'seconds')

    #save trained model weights
    torch.save(model.state_dict(), model_file_path)


if __name__ == '__main__':

    #DATASET PREPARATION 
    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    batch_size = 4

    # Path to the folder where the dataset is located or will be downloaded to (e.g. CIFAR10)
    DATASET_PATH = "./data/"
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Create checkpoint path if it doesn't exist yet
    CHECKPOINT_PATH = "./checkpoints/"
    model_file_name = "model.ckpt"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    model_file_path = os.path.join(CHECKPOINT_PATH, model_file_name)

    train_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root=DATASET_PATH, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Train dataset size: ", len(train_dataset))
    print("Test dataset size: ", len(test_dataset))

    #MODEL TRAINING 
    if not os.path.isfile(model_file_path):    
        start_time = time.time()
        train_model(train_loader)
        print('Finished model training. Elapsed time: ', time.time() - start_time, 'seconds')


    #MODEL EVALUATION on test dataset
    print(f"Loading pretrained model from {model_file_path}...")
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_file_path)) # Automatically loads the model with the saved hyperparameters

    # evaluate network accuracy on whole test dataset
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            #in case of running on GPU
            #inputs, labels = data[0].to(device), data[1].to(device)
            
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on test images: {100 * correct // total} %')

    # count predictions for each class
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    start_time = time.time()
    # no gradients needed for model validation
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            #in case of running on GPU
            #inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
                
    print("Inference time: ", time.time() - start_time, "seconds")

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
