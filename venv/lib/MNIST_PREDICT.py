import gzip
import numpy as np
import torch
import torch.nn
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

mnist_trainset = datasets.MNIST(root='/Users/jon/Desktop/MNIST_DATA', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='/Users/jon/Desktop/MNIST_DATA', train=False, download=True, transform=None)

BATCH_SIZE = 10000
FEATURE_SIZE = 784
NUM_CLASSES = 10
torch.set_printoptions(precision=4, threshold=1000,linewidth=100)

def pre_process(data):
    X = data[0][:BATCH_SIZE]
    y = data[1][:BATCH_SIZE]
    X.resize_(BATCH_SIZE, FEATURE_SIZE)
    one_hot = torch.tensor(range(0, NUM_CLASSES), dtype=torch.long)
    one_hot = one_hot.repeat(BATCH_SIZE, 1)
    y = (one_hot.t() == y).t()
    return X.double(), y.double()


def normalize(A):
    mean = torch.load('data/means.pt')
    range = 255.
    for x, m in zip(A.t(), mean):
        x.sub_(m).div_(range)
    return A


# Load the data and weights
trainset = torch.load('data/processed/test.pt')
testset = torch.load('data/processed/test.pt')
theta1 = torch.load('data/w1.pt')
theta2 = torch.load('data/w2.pt')

"""
Preprocess the training set so each example of X is a unrolled vector of that example (1 x 784) and each
example of y is a one hot vector for that example. We take the first 10000 examples
"""

X, y = pre_process(testset)
print('X[0] is', X[0].view(28, 28))
X = normalize(X)
test_image_zero, test_target_zero = mnist_testset[9839]

test_image_zero.show()

dtype = torch.double
accuracy = 0.
cnt = 0
i = 0
missclassed = []

for i in range(0, BATCH_SIZE):
    #Grab the ith example image
    example = X[i].resize_(784, 1)
    # Forward pass: compute predicted y
    example = torch.cat((torch.tensor(1, dtype=dtype).resize_(1, 1), example), 0)
    a_1_bias = example

    z_2 = theta1.mm(a_1_bias)
    a_2 = torch.sigmoid(z_2)
    a_2_bias = torch.cat((torch.tensor(1, dtype=dtype).resize_(1, 1), a_2), 0)
    z_3 = theta2.mm(a_2_bias)
    a_3 = torch.sigmoid(z_3)

    y_pred = a_3.t()

    print(i)
    print('Target_y:', y[i])
    print('Predicted_y:', y_pred)

    target = torch.argmax(y[i]).item()
    predicted = torch.argmax(y_pred).item()
    print('Target:', target)
    print('Predicted:', predicted)
    print('\n')
    if target == predicted:
        cnt += 1
    if target != predicted:
        missclassed.append(i)

print('Missclassed: ', missclassed)
accuracy = cnt / BATCH_SIZE
print('Accuracy:', accuracy)