import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from print_util import *
from utils import *
import time

def test(test_input, test_target, test_classes, model, criterion, batch_size, with_aux_loss = False, aux_loss_alpha = 0.5):
    
    with torch.no_grad():
        nb_data_errors = 0
        loss_sum = 0
        
        for inputs, targets in zip(test_input.split(batch_size),
                                  test_target.split(batch_size)):
            
            if with_aux_loss:
                outputs, output_aux = model(inputs)
                aux_loss = criterion(output_aux, targets)
                primary_loss = criterion(outputs, targets)
                loss = primary_loss +  aux_loss_alpha * aux_loss
                
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            loss_sum += loss
            _, predicted_classes = torch.max(outputs, 1)
            
            for k in range(len(inputs)):
                if targets[k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

        accuracy = (1 - (nb_data_errors / test_input.size(0))) * 100
        
        return accuracy, loss_sum.item()


def train_model(model, train_input, train_target, train_classes, test_input, test_target, test_classes, nb_epoch, batch_size, optimizer_params, logging = False, with_aux_loss = False, aux_loss_alpha = 0.5):
    
    nb_epoch, batch_size = nb_epoch, batch_size
    lr, momentum, weight_decay, gamma = optimizer_params['lr'], optimizer_params['momentum'], optimizer_params['weight_decay'], optimizer_params['gamma'] 
#     optimizer = torch.optim.Adam(model.parameters()) #, lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    
    if logging:
        log_acc_loss_header(color=Color.GREEN)
    
        train_accuracies = []
        train_losses = []
        test_accuracies = []
        test_losses = []
        start_time = time.time()
    
    
    for e in range(nb_epoch):

        for inputs, targets in zip(train_input.split(batch_size),
                                  train_target.split(batch_size)):
            
            if with_aux_loss:
                outputs, output_aux = model(inputs) 
                aux_loss = criterion(output_aux, targets)
                primary_loss = criterion(outputs, targets)
                loss = primary_loss + aux_loss_alpha * aux_loss
                
            else: 
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()  # Update the learning rate
        
        if logging:    
            train_acc, train_loss = test(train_input, train_target, train_classes, model, criterion, batch_size, with_aux_loss)
            test_acc, test_loss = test(test_input, test_target, test_classes, model, criterion, batch_size, with_aux_loss)
        
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            test_accuracies.append(test_acc)
            test_losses.append(test_loss)
        
            elapsed_time = time.time() - start_time
            log_acc_loss(e, nb_epoch, elapsed_time, train_loss, train_acc, test_loss, test_acc, persistent=False)
            
    if logging:
        print()
        return train_accuracies, train_losses, test_accuracies, test_losses


def evaluate_model(model, nb_rounds, criterion, device, batch_size, nb_epoch, optimizer_params, model_params = None, with_aux_loss = False, aux_loss_alpha = 0.5):
    
    accuracies = []
    
    for round in range(nb_rounds):
        
        # initialize new model
        if model_params != None:
        	model_evaluated = model(model_params).to(device)
        else:
        	model_evaluated = model().to(device)
        # generate new data
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_data_device(1000, device=device)
        train_input = normalize_data(train_input)
        test_input = normalize_data(test_input)
        
        train_model(model_evaluated,
                    train_input,
                    train_target,
                    train_classes,
                    None,
                    None,
                    None,
                    nb_epoch, batch_size, optimizer_params, False, with_aux_loss, aux_loss_alpha)
        
        accuracy, loss = test(test_input, test_target, test_classes, model_evaluated, criterion, batch_size, with_aux_loss, aux_loss_alpha)
        
        print("Round {i}: accuracy = {a:0.2f}% | loss = {l:0.4f}".format(i = (round + 1), a = accuracy, l = loss))
        
        accuracies.append(accuracy)
        
    return torch.FloatTensor(accuracies)


def cross_validation(model_untrained, K, train_input, train_target, train_classes, device, batch_size, nb_epoch, aux_loss_alphas):
    
    
    best_alpha = None
    best_accuracy = -1
    
    proportion = 1.0 / K
    
    # parameters you want to test
    for aux_loss_alpha in aux_loss_alphas:
        
        accuracy_sum = 0
        
        for i in range(K):
            
            model = model_untrained(aux_loss = True).to(device = device)
        
            tr_input, tr_target, tr_classes, val_input, val_target, val_classes = split_train_validation(train_input, train_target, train_classes, validation_proportion = proportion)
        
            train_model(model,
                                                                           tr_input, 
                                                                           tr_target, 
                                                                           tr_classes, 
                                                                           val_input, 
                                                                           val_target, 
                                                                           val_classes, 
                                                                           nb_epoch, 
                                                                           batch_size, 
                                                                           {'lr': 0.1, 'momentum':0.9, 'weight_decay': 0.0, 'gamma': 0.97}, 
                                                                           logging = False,
                                                                           with_aux_loss = True,
                                                                           aux_loss_alpha = aux_loss_alpha)
            
            accuracy, _ = test(val_input, val_target, val_classes, model, nn.CrossEntropyLoss(), batch_size, with_aux_loss = True, aux_loss_alpha = aux_loss_alpha)
            
            accuracy_sum += accuracy
            
        accuracy_mean = accuracy_sum / K
        
        print('aux_loss_alpha = {a} - mean accuracy = {m}'.format(a = aux_loss_alpha, m = accuracy_mean))
        
        
        if accuracy_mean > best_accuracy:
            best_accuracy = accuracy_mean
            best_alpha = aux_loss_alpha
            
    
    return best_alpha, best_accuracy