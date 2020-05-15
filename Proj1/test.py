import torch
from dlc_practical_prologue import generate_pair_sets
import torch.nn as nn
from utils import *
from utils_pipeline2 import *
import time
import models
import torch.nn.functional as F


def main():
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    BATCH_SIZE = 64
    architecture = models.SiameseNetwork
    optimizer_params = {'lr': 0.05, 'momentum':0.9, 'weight_decay': 0., 'gamma': 0.97}
    nb_epochs = 50
    nb_conv = 3
    aux_loss_alpha = 0.4
    nb_rounds = 10  # We use 10 reruns because of the high variance, reduce it to make everything faster
    
    print("Training and testing independently 10 times the model (takes a few minutes)")
    accuracies = evaluate_model(architecture, nb_conv, aux_loss_alpha, nb_rounds, nn.CrossEntropyLoss(),
                                nb_epochs, BATCH_SIZE, optimizer_params, device)
    print("The mean accuracy is: {a:0.2f}".format(a = accuracies.mean()))
    print("The accuracy standard deviation is: {s:0.4f}".format(s = accuracies.std()))
    
    
if __name__ == '__main__':
    main()