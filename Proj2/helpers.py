import math
import time
from torch import empty, tensor
import numpy as np
from modules import *
from print_util import *

def accuracy(prediction, target):
    """
    Computes the accuracy in percent between the prediction and target tensors.
    
    Args:
    prediction -- tensor of size (N, 1)
    target -- tensor of size (N, 1)
    
    Returns:
    accuracy -- the accuracy between 0% and 100%
    """
    accuracy = ((prediction-target) != 0).double()
    accuracy = (1-accuracy.mean()) * 100
    accuracy = accuracy.item()
    return accuracy

def f_score(prediction, target, alpha=0.5):
    """
    Computes the f-score measure between the prediction and target tensors.
    
    Args:
    input -- tensor of size (N, 1)
    target -- tensor of size (N, 1)
    
    Returns:
    fscore -- the fscore between 0 and 1
    """
    N = target.size(0)
    true_positive = ((target == 1) & (prediction == 1)).sum().item()
    true_negative = ((target == 0) & (prediction == 0)).sum().item()
    false_positive = ((target == 0) & (prediction == 1)).sum().item()
    false_negative = ((target == 1) & (prediction == 0)).sum().item()
    
    if ((true_positive + false_positive == 0) or 
        (true_positive + false_negative == 0) or 
        (true_positive == 0)):
        return 0
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    
    fscore = (1/(alpha*(1/precision) + (1-alpha)*(1/recall)))
    return fscore

def evaluate_model(model, input, target, logging=False):
    """
    Compute the accuracy and f-score of the given model for the given points.
    
    Args:
    model -- the pretrained model of type modules.Sequential
    input -- tensor of input points
    target -- tensor of target labels
    
    Returns:
    acc -- the accuracy
    fscore -- the f-score
    """
    prediction = model(input)
    prediction = prediction.argmax(axis=1)
    acc = accuracy(prediction, target)
    fscore = f_score(prediction, target)
    if logging:
        print("Accuracy: {0}".format(acc))
        print("F-score: {0}".format(fscore))
    return acc, fscore

def train_model(model, criterion, train_input, train_target, nb_epoch, 
                batch_size, step_size, logging=False):
    
    """Train the given model given according to the given criterion 
    (either LossMSE or LossCrossEntropy)
    
    Args:
    model -- the model of type modules.Sequential to be trained
    criterion -- the criterion of type modules.Losses
    train_input -- tensor of input
    train_target -- tensor of target
    nb_eboch -- number of eboch 
    batch_size -- size of the mini batch
    step_size -- size of the step took by sgd at each iteration
    logging -- wether or not to print on the console the evolution of the training
    
    Returns:
    accuracy -- the accuracy of the model on the training data
    fscore -- the fscore of the model on the training data
    """
    
    train_target_ = train_target.clone()
    # if the criterion is LossMSE, the train_target should be of same size
    # as the output of the model i.e. (N, 2) in the case of the Disk.
    if type(criterion) == LossMSE:
        temp = empty(size=(train_target.size(0), 2))
        for idx in range(temp.size(0)):
            if train_target[idx] == 0:
                a = 0
                b = 1.
            else:
                a = 1.
                b = 0.
            temp[idx,0] = b
            temp[idx,1] = a
        train_target = temp.clone()
        
    if logging:
        log_acc_loss_header(color=Color.GREEN)
    
        train_accuracies = []
        train_fscores = []
        train_losses = []
        start_time = time.time()
    
    for e in range(nb_epoch):        
        for inputs, targets in zip(train_input.split(batch_size),
                                  train_target.split(batch_size)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            dloss = criterion.backward(outputs, targets)
            model.backward(dloss)
            model.update_params(step_size)
            model.zero_grad()
        
        if logging:    
            prediction = model(train_input)
            prediction = prediction.argmax(axis=1)
            
            train_loss = loss
            train_acc = accuracy(prediction, train_target_)
            train_fscore = f_score(prediction, train_target_)
                        
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            train_fscores.append(train_fscore)
        
            elapsed_time = time.time() - start_time
            log_acc_loss(e, nb_epoch, elapsed_time, train_loss, train_acc, train_fscore, persistent=False)
        
    print()
    if logging:
        print("On train set:")
    acc, fscore = evaluate_model(model, train_input, train_target_, logging=logging)
    return acc, fscore
 
def generate_disk_data(nb_points = 1000, radius = 1/(math.sqrt(2*math.pi))):
    """
    Generates a training and a test set of "nb_points" points 
    sampled uniformly in [0, 1]^2 , each with a
    label, 0 if outside the disk of radius "radius" and 1 if inside.
    
    Args:
    nb_points -- the number of point to generate
    radius -- the radius of the disk
    
    Returns:
    train_input, train_target, test_input, test_target -- tensor containing the points 
    along with their labels
    """
    
    N = nb_points  
    train_input = empty(N,2).uniform_(0,1)
    test_input = empty(N,2).uniform_(0,1)
    
    train_target = ((train_input[:,0]**2 + train_input[:,1]**2) < radius**2).long()
    test_target = ((test_input[:,0]**2 + test_input[:,1]**2) < radius**2).long()
    return train_input, train_target, test_input, test_target


def plot_classifier(model, train_input, train_target, show=True):
    """
    Plot the classifier defined by the model
    
    Args:
    train_input -- tensor of input points
    train_target -- tensor of target labels
    """
    h = 0.02
    x_min, x_max = train_input[:, 0].min(), train_input[:, 0].max()
    y_min, y_max = train_input[:, 1].min(), train_input[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid_points = tensor(np.c_[xx.ravel(), yy.ravel()]).float()
    output = tensor(np.zeros((grid_points.shape[0],2))).double()
    output = model.forward(grid_points)
    output = np.argmax(output, axis=1)
    output = output.reshape(xx.shape)

    N=1000
    fig = plt.figure()
    plt.contourf(xx, yy, output, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(train_input[:N, 0], train_input[0:N, 1], c=train_target[0:N], s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig("classifier.png")
    if show==True:
        plt.show()