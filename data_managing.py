import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

## get dataset
train = datasets.MNIST("", train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train=False, download=True, transform = transforms.Compose([transforms.ToTensor()]))

## make sets with batchsize 10
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

for data in trainset:
    print(data)
    break

## draw a image and print its true value
X, y = data[0][0], data[1][0] 
plt.imshow(X.view(28,28))
plt.show()
print(y)

## cgeck if it is a calibrated trainingset
total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xi, yi = data
    for y in yi:
        counter_dict[int(y)] += 1

print(counter_dict)