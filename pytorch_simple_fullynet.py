# Imports
import torch # entire library
import torch.nn as nn # all nn modules (linear, cnn, loss functions, etc
import torch.optim as optim # optimization algorithms (stochastic gradient descent, etc)
import torch.nn.functional as F # activation functions (w/o parameters), also in nn package (relu, tanh, etc)
from torch.utils.data import DataLoader # easier dataset management, helps create minibacthes for training
from torch.utils.data.dataset import random_split # used to split dataset into training and validation
import torchvision.datasets as datasets # helps import datasets
import torchvision.transforms as transforms # transformations to perform on dataset

# Create Fully Connected Network
class NN(nn.Module): # inherit from nn.Module
    def __init__(self, input_size=784, num_class=10): # MNIST has 28x28 = 784 images, 10 classes for digits
        super(NN, self).__init__() # calls initialization method from nn.Module
        
        # Hidden Layer 1
        self.fc1 = nn.Linear(input_size, 50)
        # Hidden Layer 2
        self.fc2 = nn.Linear(50, num_class)
        
    def forward(self, x):
        # Perform layers created in init
        x = torch.flatten(x, start_dim=1) # flatten image to vector
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# model = NN(784, 10) # 10 for num of digits
# x = torch.randn(64, 784) # 64 for minibatch size
# print(model(x).shape) # returns [64, 10]


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # runs of gpu or cpu

# Hyperparameters
input_size = 784 # MNIST dataset size
num_classes = 10 # for each digit
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor() #, download=True
)

test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor()
)

# Create validation set
torch.manual_seed(1)
train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])


# Create data loaders
train_loader = DataLoader( # to make sure we don't have exact images in a batch every epoch
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)

val_loader = DataLoader(
    dataset=val_dataset, batch_size=batch_size, shuffle=False
)

# Check accuracy on training & test to see how good our model is

def compute_accuracy(loader, model):
    # if loader == train_loader:
    #     print("Checking accuracy on training data")
    # elif loader == val_loader:
    #     print("Checking accuracy on validation data")
    # else:
    #     print("Checking accuracy on test data")

    num_correct = 0.0
    num_samples = 0
    
    model = model.eval()
    
    for idx, (features, labels) in enumerate(loader):
        with torch.no_grad(): # don't calculate gradients
            features = features.to(device=device)
            labels = labels.to(device=device)
            logits = model(features)
        
        predictions = torch.argmax(logits, dim=1) # get index of max value in row
        
        num_correct += (predictions == labels).sum()
        num_samples += predictions.size(0)
        
    acc = (num_correct)/(num_samples)
    return acc


# Initialize network
model = NN(input_size=input_size, num_class=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # or use F.cross_entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # adam optimizer
# optimizer = optim.SGD(model.parameters(), lr=learning_rate) # stochastic gradient descent optimizer

# Train network

loss_list = []
train_acc_list, val_acc_list, test_acc_list = [], [], []
for epoch in range(num_epochs): # an epoch is like a generation, where the network/model has seen all images
    for batch_idx, (features, labels) in enumerate(train_loader): # goes through each batch in dataloader, with index
        # Get data to cuda if possible
        features = features.to(device=device)
        labels = labels.to(device=device)
        
        # print(features.shape) # prints torch.Size([64, 1, 28, 28]) --> ([num images in batch, number of channels (B&W o/w 3 with RGB), height, width of image])
    
        # Get correct shape, by unrolling matrix to single vector
        features = features.reshape(features.shape[0], -1) # (28, 28) --> 784
        
        # Forward
        logits = model(features) # Logits are the raw scores output by the last layer of the model.
        loss = F.cross_entropy(logits, labels) # or criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad() # set gradients to zero before backpropagation
        loss.backward() # backpropagation
        
        # Gradient descent or adam step
        optimizer.step() # update weights depending on gradients calculated in loss.backward()
        
        if not batch_idx % 250:
            ### LOGGING
            print(
                f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                f" | Train Loss: {loss:.2f}"
            )
        loss_list.append(loss.item())
        
    train_acc = compute_accuracy(train_loader, model)
    val_acc = compute_accuracy(train_loader, model)
    test_acc = compute_accuracy(test_loader, model)
        
    print(f"Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}% | Test Acc {test_acc*100:.2f}%")
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    test_acc_list.append(test_acc)
