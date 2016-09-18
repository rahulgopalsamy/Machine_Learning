import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
import pickle

start_time = time.time()
np.set_printoptions(threshold=np.inf)


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon;
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sig = 1 / (1 + np.exp(-z))
    return sig



trainlg = None


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    # print mat

    # Pick a reasonable size for validation data

    # Initialising the arrays
    train_data = np.array([], dtype=np.float).reshape(0, 784)
    validation_data = np.array([], dtype=np.float).reshape(0, 784)
    test_data = np.array([], dtype=float).reshape(0, 784)
    train_label = np.array([])
    validation_label = np.array([])
    test_label = np.array([])
    train_label = np.array([])

    # Loading data set
    for i in range(0, 10):
        temp_test_label = np.zeros((mat.get('test' + str(i)).astype(np.float).shape[0], 1))
        temp_test_label.fill(i)

        temp_train_label = np.empty(np.random.permutation(mat.get('train' + str(i)).astype(np.float)).shape[0])
        temp_train_label.fill(i)

        # Initially when array is empty fill the array with first set of values
        # Stack the values from the next iteration
        if (test_data.size == 0):
            test_data = mat.get('test' + str(i)).astype(np.float)
        else:
            test_data = np.vstack([test_data, mat.get('test' + str(i)).astype(np.float)])

        test_label = np.append(test_label, temp_test_label)
        # print "testlabel",test_label.shape

        if (validation_data.size == 0):
            validation_data = np.random.permutation(mat.get('train' + str(i)).astype(np.float))[:1000]
        else:
            validation_data = np.vstack(
                [validation_data, np.random.permutation(mat.get('train' + str(i)).astype(np.float))[:1000]])

        validation_label = np.append(validation_label, temp_train_label[:1000])

        if (train_data.size == 0):
            train_data = np.random.permutation(mat.get('train' + str(i)).astype(np.float))[1000:]
        else:
            train_data = np.vstack(
                [train_data, np.random.permutation(mat.get('train' + str(i)).astype(np.float))[1000:]])

        train_label = np.append(train_label, temp_train_label[1000:])

    # import pdb
    # pdb.set_trace()
    # Normalise
    train_data /= 255
    validation_data /= 255
    test_data /= 255

    # Feature Selection
    full_data = np.array(np.vstack((train_data, validation_data, test_data)))
    N = np.all(full_data == full_data[0, :], axis=0)
    full_data = full_data[:, ~N]

    train_data = full_data[0:train_data.shape[0], :]
    validation_data = full_data[train_data.shape[0]: train_data.shape[0] + validation_data.shape[0], :]
    test_data = full_data[
                train_data.shape[0] + validation_data.shape[0]: train_data.shape[0] + validation_data.shape[0] +
                                                                test_data.shape[0], :]


    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape(n_hidden, (n_input + 1))
    w2 = params[(n_hidden * (n_input + 1)):].reshape(n_class, (n_hidden + 1))
    obj_val = 0

    # feed forward input to hidden
    # biasing the training data and calculating Zj
    # Eqn 1 and 2
    bias = np.zeros(training_data.shape[0])
    bias.fill(1)
    training_data = np.column_stack([training_data, bias])  # X1
    aj = np.dot(training_data, w1.T)
    zj = sigmoid(aj)

    # Feed forward hidden to output
    # Biasing the Z value and calculating the value of ol
    # Adding bias to zj
    zj2 = np.column_stack([zj, bias])
    
    # Feed forward hidden to output
    a2j = np.dot(zj2, w2.T)
    ol = sigmoid(a2j)

    # Generating yl
    yl = np.zeros((50000, 10))
    for i in range(0, 50000):
        yl[i, training_label[i]] = 1


    delta = np.multiply(yl - ol, (1 - ol) * ol)


    # Back propogation
    # input to hidden
    # Removing the bias node from the w2 and calculating the gradient value for w1

    w211 = w2[:, range(0, w2.shape[1] - 1)]
    zj11 = zj2[:, range(0, zj2.shape[1] - 1)]

    temp1 = -((1 - zj11) * zj11)
    temp2 = (np.dot(delta, w211))
   
    temp = np.multiply(temp1, temp2)


    grad_w1 = np.dot(temp.T, training_data)


    grad_w1 = ((grad_w1 + lambdaval * w1) / training_data.shape[0])

    # Calculating the gradient value for w2
    grad_w2 = -(np.dot(delta.T, zj2))
    grad_w2 = (grad_w2 + lambdaval * w2) / training_label.shape[0]

    # Error value
    J = (np.square(yl - ol)) / (2)
    J = np.sum(J)

    obj_val += (lambdaval * (np.sum(np.square(w1)) + np.sum(np.square(w2))) / (2 * training_data.shape[0])) + J
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predictest labels"""
    labels = np.zeros((data.shape[0], 1))
    
    #Feedforward
    bias = np.zeros(data.shape[0])
    bias.fill(1)
    training_data = np.column_stack([data, bias])  # X1
    aj = np.dot(training_data, w1.T)
    zj = sigmoid(aj)
    zj = np.column_stack([zj, bias])

    # feed forward hidden to output
    a2j = np.dot(zj, w2.T)
    ol = sigmoid(a2j)

    labels = ol.argmax(axis=1)
    # labels = np.argmax(ol, axis=1)
    # print ("labels's shape",labels.shape)
    # print ("label",labels)
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 20;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0.3;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

pickle.dump([w1,w2,n_hidden,lambdaval],open("params.pickle","wb"))
predictest_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predictest_label == train_label).astype(float))) + '%')

predictest_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predictest_label == validation_label).astype(float))) + '%')

predictest_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predictest_label == test_label).astype(float))) + '%')
print("Seconds %s" % (time.time() - start_time))
