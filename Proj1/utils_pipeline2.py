import torch
from utils import *
import torch.nn as nn
import time

def test(test_input, test_target, test_classes, model, criterion, batch_size, aux_loss_alpha=0.0):
    """
    This method tests the model passed as argument on the testing data
    """
    model.eval()  # Switch to eval mode in case we use an architecture that requires it
    with torch.no_grad():
        nb_final_errors = 0
        nb_digit_errors = 0
        loss_sum = 0
        
        for inputs, classes, targets in zip(test_input.split(batch_size), 
                                            test_classes.split(batch_size), 
                                            test_target.split(batch_size)):

            classes1, classes2 = classes[:, 0], classes[:, 1]
            inputs1, inputs2 = separate_channels(inputs)
            outputs1, outputs2 = model.digit_pred(inputs1), model.digit_pred(inputs2)
            loss_digit = criterion(outputs1, classes1) + criterion(outputs2, classes2)
            loss_sum += aux_loss_alpha * loss_digit
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)

            for k in range(len(inputs)):
                if classes1[k] != predicted1[k]:
                    nb_digit_errors += 1
                if classes2[k] != predicted2[k]:
                    nb_digit_errors += 1
            
            outputs = model(inputs)
            loss_final = criterion(outputs, targets)
            loss_sum += loss_final
            _, predicted = torch.max(outputs, 1)
            
            for k in range(len(inputs)):
                if targets[k] != predicted[k]:
                    nb_final_errors += 1

        final_acc = (1 - (nb_final_errors / test_input.size(0))) * 100
        digit_acc = (1 - (nb_digit_errors / (test_input.size(0) * 2))) * 100
        
        return final_acc, digit_acc, loss_sum.item()
    
    
def train_model(model, train_input, train_target, train_classes, test_input, test_target, test_classes,
                nb_epochs, batch_size, optimizer_params, logging = False, aux_loss_alpha=0.0):
    """
    This method is used to train the model passed as argument
    """
    lr, momentum, weight_decay, gamma = optimizer_params['lr'], optimizer_params['momentum'], optimizer_params['weight_decay'], optimizer_params['gamma']    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
    criterion = nn.CrossEntropyLoss()
    
    if logging:
        log_acc_loss_header(color=Color.GREEN, aux=True)
    
        train_final_accuracies = []
        train_digit_accuracies = []
        train_losses = []
        test_final_accuracies = []
        test_digit_accuracies = []
        test_losses = []
        start_time = time.time()
    
    for e in range(nb_epochs):
        model.train()  # Switch to train mode in case we use an architecture that requires it
        for inputs, targets, classes in zip(train_input.split(batch_size),
                                            train_target.split(batch_size),
                                            train_classes.split(batch_size)):
            
            inputs1, inputs2 = separate_channels(inputs)
            outputs1, outputs2 = model.digit_pred(inputs1), model.digit_pred(inputs2)
            loss_digit = criterion(outputs1, classes[:, 0]) + criterion(outputs2, classes[:, 1])
            loss = aux_loss_alpha * loss_digit
                
            outputs = model(inputs)
            loss_final = criterion(outputs, targets)
            loss += loss_final
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        scheduler.step()  # Update the learning rate
        
        if logging:
            train_final_acc, train_digit_acc, train_loss = test(train_input, train_target, train_classes, model, criterion, batch_size, aux_loss_alpha=aux_loss_alpha)
            test_final_acc, test_digit_acc, test_loss = test(test_input, test_target, test_classes, model, criterion, batch_size, aux_loss_alpha=aux_loss_alpha)
        
            train_final_accuracies.append(train_final_acc)
            train_digit_accuracies.append(train_digit_acc)
            train_losses.append(train_loss)
            
            test_final_accuracies.append(test_final_acc)
            test_digit_accuracies.append(test_digit_acc)
            test_losses.append(test_loss)
        
            elapsed_time = time.time() - start_time
            log_acc_loss_aux(e, nb_epochs, elapsed_time, train_loss, train_final_acc, train_digit_acc, test_loss, test_final_acc, test_digit_acc, persistent=False)
            
    if logging:
        print()
        return train_final_accuracies, train_digit_accuracies, train_losses, test_final_accuracies, test_digit_accuracies, test_losses
    

def evaluate_model(architecture, nb_conv, aux_loss_alpha, nb_rounds, criterion, nb_epochs, batch_size, optimizer_params, device):
    """
    This method is used to evaluate the model passed as argument
    """
    accuracies = []
    log_evaluate_header(color=Color.GREEN)
    
    for round in range(nb_rounds):
        # initialize new model
        model_evaluated = architecture(nb_conv=nb_conv, final_bias=True).to(device=device)
        # generate new data
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_data_device(1000, device=device)
        train_input = normalize_data(train_input)
        test_input = normalize_data(test_input)
        
        train_model(model_evaluated,
                    train_input, train_target, train_classes,
                    None, None, None,
                    nb_epochs, batch_size, 
                    optimizer_params, aux_loss_alpha=aux_loss_alpha)
        
        accuracy, _, loss = test(test_input, test_target, test_classes, model_evaluated, criterion, batch_size, aux_loss_alpha=aux_loss_alpha)
        log_evaluate(round, nb_rounds, accuracy, loss, persistent=True)
        accuracies.append(accuracy)
        
    return torch.FloatTensor(accuracies)


def cross_validation(architecture, K, train_input, train_target, train_classes, device, batch_size, nb_epoch, aux_loss_alphas, optimizer_params):
    """
    This methods performs a cross validation and returns the best alpha used with auxiliary loss
    """
    best_alpha = None
    best_accuracy = -1
    
    proportion = 1.0 / K
    
    # parameters you want to test
    for aux_loss_alpha in aux_loss_alphas:
        accuracy_sum = 0
        
        for i in range(K):
            model = architecture(nb_conv=2).to(device = device)
            tr_input, tr_target, tr_classes, val_input, val_target, val_classes = split_train_validation(train_input, train_target, train_classes, validation_proportion = proportion)
            train_model(model, tr_input, tr_target, tr_classes, val_input, val_target, val_classes, 
                       nb_epoch, batch_size, optimizer_params, 
                       logging=False, aux_loss_alpha=aux_loss_alpha)
            
            accuracy, _, _ = test(val_input, val_target, val_classes, model, nn.CrossEntropyLoss(), batch_size, aux_loss_alpha=aux_loss_alpha)
            accuracy_sum += accuracy
            
        accuracy_mean = accuracy_sum / K
        
        print('Aux loss alpha = {a} => Mean accuracy = {m}'.format(a = aux_loss_alpha, m = accuracy_mean))
        
        if accuracy_mean > best_accuracy:
            best_accuracy = accuracy_mean
            best_alpha = aux_loss_alpha    
    
    return best_alpha, best_accuracy

