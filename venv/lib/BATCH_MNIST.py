"""Simple 3 layer MLP for training on the MNIST dataset.
Notes
-----
This Neural Network is trained using Mini-Batch gradient descent on
a training sample size of 60000 handwritten digits. In this example,
the sigmoid activation is used along with the cross-entropy loss function.
Weights are optimized using the Adam optimization routine
"""
import numpy as np
import torch
import torchvision.datasets as datasets

# Load the datasets to visualize using torchvision.datasets
mnist_trainset = datasets.MNIST(root='/Users/jon/Desktop/MNIST_DATA', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='/Users/jon/Desktop/MNIST_DATA', train=False, download=True, transform=None)

# Load the training data.
train_set = torch.load('data/processed/training.pt')
"""
Define hyper-parameters. EPOCHS is the number of passes through 
the training data. BATCH_SIZE is the training examples.
FEATURE_SIZE is 784 since 28 x 28 image is unrolled into a feature vector of
length 784. MINI_BATCH_SIZE is number of training examples processed
between weight updates. 
"""
EPOCHS = 20
BATCH_SIZE = 60000
FEATURE_SIZE = 784
HIDDEN_LAYER_SIZE = 256
NUM_CLASSES = 10
MINI_BATCH_SIZE = 64
device = torch.device("cpu")
dtype = torch.double
torch.set_printoptions(precision=12, threshold=800,linewidth=1000)


class Network():
    """
    Multi-Layer-Perceptron model with 1 hidden layer.
    Parameters
    ----------
    theta1 - the weight matrix used to calculate the hidden layer
    theta2 - the weight matrix used to calculate outputs
    l2_reg - a regularization constant to multiply the sum of squares
    of weights in the cost function to prevent overfitting
    """
    def __init__(self, theta1, theta2, l2_reg=0):
        self.theta1 = theta1
        self.theta2 = theta2
        self.l2_reg = l2_reg

    def __forward(self, X):
        """
        Forward propagation to compute the predicted labels.
        :return: the prediction labels, and the hidden layer
        """
        a_1_bias = X
        z_2 = a_1_bias.mm(self.theta1.t())
        a_2 = torch.sigmoid(z_2)
        a_2_bias = cat_ones(a_2)
        z_3 = a_2_bias.mm(self.theta2.t())
        y_pred = torch.sigmoid(z_3)
        return y_pred, a_2, a_2_bias

    def back_prop(self, X, y, minibatch_size):
        """
        Backwards propagation to compute the gradients of the weight
        matrices w.r.t the cross-entropy loss function.
        :return: the gradients
        """
        a_1_bias = X
        y_pred, a_2, a_2_bias = self.__forward(a_1_bias)

        delta_3 = (y_pred - y)
        delta_2 = (delta_3.mm(theta2[:, 1:])) * (a_2 * (1 - a_2))
        grad1 = delta_2.t().mm(a_1_bias)
        grad2 = delta_3.t().mm(a_2_bias)

        # Update gradient with regularization
        grad1[:, 1:] += (self.l2_reg * self.theta1[:, 1:])
        grad1 /= minibatch_size
        grad2[:, 1:] += (self.l2_reg * self.theta2[:, 1:])
        grad2 /= minibatch_size
        return grad1, grad2

    def J(self, X, y, l2_reg):
        """
        The logistic cost for the model with regularization. The theta is taken in
        its unrolled form. Lambda value is the L2 regularization constant.
        :return: the cost
        """
        y_pred = self.__forward(X)[0]
        m = len(y)
        cost_matrix = y.mm(y_pred.log().t()) + (1 - y).mm((1 - y_pred).log().t())
        cost = cost_matrix.diag().sum() / (-m)
        cost += (l2_reg / (2 * m)) * (self.theta1[:, 1:].pow(2).sum().sum())
        cost += (l2_reg / (2 * m)) * (self.theta2[:, 1:].pow(2).sum().sum())
        return cost.item()

    def accuracy(self, X, y):
        """
        Computes accuracy of the model on (X, y) examples.
        :return: the accuracy
        """
        y_pred = self.__forward(X)[0]
        return (torch.sum(torch.argmax(y_pred, 1) == torch.argmax(y, 1)).item()) * 100. / MINI_BATCH_SIZE

    def check_grad(self, X, y):
        """
        Computes a numerical approximation to the gradient of each weight in
        the theta matrices. TODO: Update cost function so it accepts
        unrolled parameters.
        :return: the gradient approximation
        """
        theta = unroll(self.theta1, self.theta2)
        epsilon = 1e-4
        grad_approx = torch.zeros(theta.numel(), dtype=dtype)
        perturb = torch.zeros(theta.numel(), dtype=dtype)
        for i in range(0, theta.numel()):
            perturb[i] = epsilon
            loss1 = self.J(theta - perturb, X, y)
            loss2 = self.J(theta + perturb, X, y)
            grad_approx[i] = (loss2 - loss1) / (2 * epsilon)
            perturb[i] = 0
        return grad_approx

def shuffle(X, y):
    """
    Shuffles the examples in the data randomly.
    Note that each example occupies a row.
    """
    rand_idx = torch.randperm(BATCH_SIZE)
    return X[rand_idx], y[rand_idx]

def pre_process(data):
    """
    Takes the first BATCH_SIZE examples from the dataset and processes
    the input and labels. Specifically, the labels are transformed
    into one-hot vectors where row i corresponds to the one-hot
    vector associated with example i. X is a matrix where each row i
    corresponds to the unrolled feature vector of example i.
    """
    X = data[0][:BATCH_SIZE]
    y = data[1][:BATCH_SIZE]
    X.resize_(BATCH_SIZE, FEATURE_SIZE)
    one_hot = torch.tensor(range(0, NUM_CLASSES), dtype=torch.long)
    one_hot = one_hot.repeat(BATCH_SIZE, 1)
    y = (one_hot.t() == y).t()
    return X.double(), y.double()


def initialize_weights(Theta):
    """
    Initialize Theta weights for layer l with values chosen uniformly in
    [-epsilon_init, epsilon_init], where epsilon_init is given by
    sqrt(6) / sqrt(l_in + l_out) (Read more at http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
    L_in is the number of activations in layer l and l_out is the number of activations in layer l + 1.
    Also, a bias node of 1 will be added to each input later.
    """
    l_out, l_in = Theta.size()
    epsilon_init = np.sqrt(6 / (l_in + l_out))
    Theta *= (2 * epsilon_init)
    Theta -= epsilon_init
    return Theta


def normalize(A):
    """
    Mean normalization and feature scaling using the range of 255.
    """
    mean = torch.mean(A.double(), 0)
    torch.save(mean, 'data/means.pt')
    # std = torch.std(A.float(), 0)
    # std[std == 0] = 1
    range = 255.
    # for x, m, s in zip(A.t(), mean, std):
    #     x.sub_(m).div_(s)
    for x, m in zip(A.t(), mean):
        x.sub_(m).div_(range)
    # A /= range
    return A


def cat_ones(A):
    """
    Concatenates a column of ones to a matrix.
    """
    ones = (torch.tensor([1] * A.size(0), dtype=dtype)).resize_(A.size(0), 1)
    return torch.cat((ones, A), 1)


def unroll(p1, p2):
    """
    Flatten and concatenate the individual theta matrices.
    This is similar to ravel in numpy.
    """
    return torch.cat((p1.reshape(theta1.numel()), p2.reshape(theta2.numel())))


def reshape(p):
    """
    Returns the unrolled parameters in their original shape.
    """
    p1 = p[0: HIDDEN_LAYER_SIZE * (FEATURE_SIZE + 1)]
    p1 = p1.reshape(HIDDEN_LAYER_SIZE, FEATURE_SIZE + 1)
    p2 = p[HIDDEN_LAYER_SIZE * (FEATURE_SIZE + 1):].reshape(NUM_CLASSES, HIDDEN_LAYER_SIZE + 1)
    return p1, p2


def get_progress_bar(end, BATCH_SIZE):
    """
    Returns a progress bar String which updates as training.
    progress is made.
    """
    s = '['
    c1cnt = int((end / BATCH_SIZE) * 30)
    s += '=' * c1cnt
    c2cnt = 30 - (c1cnt + 1)
    if c1cnt != 0:
        s += '>'
    else:
        c2cnt += 1
    s += '.' * c2cnt
    s += ']'
    return s


X, y = pre_process(train_set)
X = normalize(X)
X = cat_ones(X)

"""
To help visualize the images and their data, show the digit image.
Target is its class. Example number is ex.
"""
ex = 0
train_image_zero, train_target_zero = mnist_trainset[0]
# train_image_zero.show()
# print(train_target_zero.item())

"""
Specifies the network architecture.
N is batch size; D_in is input dimension;
H is hidden dimension; D_out is output dimension.
"""
N, D_in, H, D_out = BATCH_SIZE, FEATURE_SIZE, HIDDEN_LAYER_SIZE, NUM_CLASSES

"""
Weight initialization with bias unit weights included.
Hyper-parameters involved in learning set here.
"""
theta1 = torch.rand(H, D_in + 1, device=device, dtype=dtype)
theta2 = torch.rand(D_out, H + 1, device=device, dtype=dtype)
theta1 = initialize_weights(theta1)
theta2 = initialize_weights(theta2)
theta = unroll(theta1, theta2)
LEARNING_RATE = .001
LAMBDA_VAL = .002
BETA1 = .9
BETA2 = .999

nn = Network(theta1, theta2)

"""
Variables used in Adam optimization routine.
Momentum is a moving average of the gradients and
dampens oscillations while descending the
gradient. Rms keeps track of the moving average
of the square of the gradient and is used to optimally
modify the learning rate of each individual weight.
"""
momentum1 = torch.zeros(theta1.size(), dtype=dtype)
momentum2 = torch.zeros(theta2.size(), dtype=dtype)
rms1 = torch.zeros(theta1.size(), dtype=dtype)
rms2 = torch.zeros(theta2.size(), dtype=dtype)

"""
t is number of iterations, DISPLAY_FREQ is the number of
iterations before updating information 
"""
DISPLAY_FREQ = 10
t = 0
cost = 0
for epoch in range(EPOCHS):
    X, y = shuffle(X, y)
    start = 0
    end = MINI_BATCH_SIZE
    while end < BATCH_SIZE:
        X_sub = X[start: end]
        y_sub = y[start: end]
        if t % DISPLAY_FREQ == 0:
            acc = (nn.accuracy(X_sub, y_sub))
        cost += nn.J(X_sub, y_sub, l2_reg=LAMBDA_VAL)
        grad1, grad2 = nn.back_prop(X_sub, y_sub, MINI_BATCH_SIZE)

        # Adam
        momentum1 = BETA1 * momentum1 + (1 - BETA1) * grad1
        momentum2 = BETA1 * momentum2 + (1 - BETA1) * grad2
        rms1 = BETA2 * rms1 + (1 - BETA2) * grad1.pow(2)
        rms2 = BETA2 * rms2 + (1 - BETA2) * grad2.pow(2)

        # # Bias Correction
        # momentum1 /= (1 - BETA1 ** (t + 1))
        # momentum2 /= (1 - BETA1 ** (t + 1))
        # rms1 /= (1 - BETA2 ** (t + 1))
        # rms2 /= (1 - BETA2 ** (t + 1))

        # Weight update
        nn.theta1 -= LEARNING_RATE * (momentum1 / (rms1.sqrt() + 10e-8))
        nn.theta2 -= LEARNING_RATE * (momentum2 / (rms2.sqrt() + 10e-8))

        # Vanilla Gradient Descent
        # nn.theta1 -= LEARNING_RATE * grad1
        # nn.theta2 -= LEARNING_RATE * grad2

        start = end
        end += MINI_BATCH_SIZE
        if t % DISPLAY_FREQ == 0:
            cost /= DISPLAY_FREQ
            progress_bar = get_progress_bar(end, BATCH_SIZE)
            print('EPOCH {}: {}/{} {} cost: {} - acc: {}'.format(epoch, end, BATCH_SIZE, progress_bar,cost, acc))
            cost = 0
        t += 1

torch.save(nn.theta1, 'data/w1.pt')
torch.save(nn.theta2, 'data/w2.pt')