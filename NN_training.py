import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

## image sizes are 28x28
## get dataset
train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

## load batches of size 10 into sets
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset =  torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64) #input(28x28 = 784), output(next layer size)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) #input(64 from before), output(0,1,...,9)

    def forward(self, x): 
        x = F.relu(self.fc1(x)) # F.relu is the activation function
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

# initialize Network
net = Net()

#X = torch.rand((28, 28)) # make random image
#X = X.view(-1, 28*28) # flatten data because input is array of size 784, -1: any size of batch
#output = net(X)
#print(output)


optimizer = optim.Adam(net.parameters(), lr=0.001)

for epoch in range(3): # 3 full passes over the data
    for data in trainset:  # data is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        net.zero_grad()  # sets gradients to 0 before loss calc. Don't want to add up all gradients, just per batch
        output = net(X.view(-1,784))  # pass in the reshaped batch 
        loss = F.nll_loss(output, y)  # calc and grab the loss value, nll because we have a number. if y was one hot vector the mean square error
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 

## calculate the Accuracy
correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 784))
        for idx, i in enumerate(output): #idx is index
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct/total*100, 2))

import matplotlib.pyplot as plt

plt.imshow(X[0].view(28,28))
plt.show()

print(torch.argmax(net(X[0].view(-1,784))[0]))
#net(X[0].view(-1,784)) returns a list of predictions