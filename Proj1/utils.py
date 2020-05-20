from dlc_practical_prologue import generate_pair_sets
import torch
import matplotlib.pyplot as plt

def generate_data_device(n, device='cpu'):
    """
    Generates the data for the project and send it to the device passed as argument
    """
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n)
    train_input = train_input.to(device=device)
    train_target = train_target.to(device=device)
    train_classes = train_classes.to(device=device)
    test_input = test_input.to(device=device)
    test_target = test_target.to(device=device)
    test_classes = test_classes.to(device=device)
    return train_input, train_target, train_classes, test_input, test_target, test_classes

def normalize_data(tensor):
    """
    Normalizes the tensor passed as argument
    """
    mu, std = tensor.mean(), tensor.std()
    tmp = tensor.sub(mu).div(std)
    return tmp

def separate_channels(x):
    """
    Separates the two channel of the tensor passed as argument into two different tensors
    """
    x1 = x[:, 0:1, :, :]
    x2 = x[:, 1:2, :, :]
    return x1, x2

def split_train_validation(train_input, train_target, train_classes, validation_proportion = 0.2):
    """
    Splits the train tensors passed as argument into new train and validations tensors with 'validation_proportion' of the data
    """
    index_permutation = torch.randperm(train_input.size(0))
    split = int(0.2 * train_input.size(0))

    validation_index = index_permutation[:split]
    training_index = index_permutation[split:]

    validation_input = train_input[validation_index]
    validation_target = train_target[validation_index]
    validation_classes = train_classes[validation_index]

    train_input = train_input[training_index]
    train_target = train_target[training_index]
    train_classes = train_classes[training_index]
    
    return train_input, train_target, train_classes, validation_input, validation_target, validation_classes

class Color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    BRIGHT_GREEN = '\033[92m'
    GREEN = '\033[32m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    GRAY = '\033[90m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_test(b, batch_length, batch_time, loss, top1, persistent=True, color='', title='Test'):
    """
    Prints the results when testing
    """
    print('\r' + color + title.ljust(14) +
          '[{0}/{1}]'.format(b+1, batch_length).ljust(14) +
          '{batch_time.sum:.0f}s'.format(batch_time=batch_time).ljust(7) +
          '{loss.val:.4f} ({loss.avg:.4f})'.format(loss=loss).ljust(25) +
          '{top1.val:.3f} ({top1.avg:.3f})'.format(top1=top1).ljust(22) + Color.END,
          end='\n' if persistent else '')


def log_acc_loss_header(color='', aux=False):
    """
    Prints the accuracy and loss header
    """
    if aux:
        print(color + 'Epoch'.ljust(12) + 'Time'.ljust(8) + 'Train loss'.ljust(15) +
          'Train acc'.ljust(15) + 'Train digit acc'.ljust(20) + 
              'Test loss'.ljust(15) + 'Test acc'.ljust(15) + 'Test digit acc'.ljust(20) + Color.END)
    else:
        print(color + 'Epoch'.ljust(12) + 'Time'.ljust(8) + 'Train loss'.ljust(15) +
          'Train accuracy'.ljust(20) + 'Test loss'.ljust(15) + 'Test accuracy'.ljust(20) + Color.END)


def log_acc_loss(e, nb_epoch, time, train_loss, train_acc, test_loss, test_acc, color='', persistent=True):
    """
    Prints the information related to training
    """
    print('\r' + color +
          '[{0}/{1}]'.format(e + 1, nb_epoch).ljust(12) +
          '{time:.0f}s'.format(time=time).ljust(8) +
          '{0:.4f}'.format(train_loss).ljust(15) +
          '{0:.4f}'.format(train_acc).ljust(20) +
          '{0:.4f}'.format(test_loss).ljust(15) +
          '{0:.4f}'.format(test_acc).ljust(20) +
          Color.END,
          end='\n' if persistent else '')
    
    
def log_acc_loss_aux(e, nb_epoch, time, train_loss, train_acc, train_digit_acc, test_loss, test_acc, test_digit_acc, color='', persistent=True):
    """
    Prints the information related to training with auxiliary loss
    """
    print('\r' + color +
          '[{0}/{1}]'.format(e + 1, nb_epoch).ljust(12) +
          '{time:.0f}s'.format(time=time).ljust(8) +
          '{0:.4f}'.format(train_loss).ljust(15) +
          '{0:.4f}'.format(train_acc).ljust(15) +
          '{0:.4f}'.format(train_digit_acc).ljust(20) +
          '{0:.4f}'.format(test_loss).ljust(15) +
          '{0:.4f}'.format(test_acc).ljust(15) +
          '{0:.4f}'.format(test_digit_acc).ljust(20) +
          Color.END,
          end='\n' if persistent else '')
    
    
def log_evaluate_header(color=''):
    """
    Prints header when evaluating
    """
    print(color + 'Round'.ljust(12) + "Test accuracy".ljust(20) + "Test loss".ljust(15) + Color.END)
    
    
def log_evaluate(r, nb_round, test_acc, test_loss, color='', persistent=True):
    """
    Prints the result when evaluating
    """
    print('\r' + color + '[{0}/{1}]'.format(r + 1, nb_round).ljust(12) +
          '{0:.4f}'.format(test_acc).ljust(20) + 
          '{0:.4f}'.format(test_loss).ljust(15) +
          Color.END,
          end='\n' if persistent else '')


def plot_acc_loss(train_accuracies, train_losses, test_accuracies, test_losses):
    """
    Plots the training results
    """
    n = len(train_accuracies)
    major_ticks = list(range(0, n, 10))
    minor_ticks = list(range(0, n, 1))

    fig, axs = plt.subplots(2, dpi=240, figsize=(15, 12))
    axs[0].plot(train_accuracies, color='Blue')
    axs[0].plot(test_accuracies, color='Red')
    axs[0].set_title("Accuracy")
    axs[0].set(xlabel='Training epochs', ylabel='Accuracy (%)')
    axs[0].grid()
    axs[0].legend(['Train set', 'Test set'])
    axs[0].set_xlim(left=0)
    axs[0].set_xlim(right=n - 1)
    axs[0].set_xticks(major_ticks)
    axs[0].set_xticks(minor_ticks, minor=True)
    axs[0].grid(which='minor', alpha=0.3)
    axs[0].grid(which='major', alpha=0.7)

    axs[1].plot(train_losses, color='Blue')
    axs[1].plot(test_losses, color='Red')
    axs[1].set_title("Cross-Entropy Loss")
    axs[1].set(xlabel='Training epochs', ylabel='Loss')
    axs[1].grid()
    axs[1].legend(['Train set', 'Test set'])
    axs[1].set_xticks(major_ticks)
    axs[1].set_xticks(minor_ticks, minor=True)
    axs[1].grid(which='minor', alpha=0.3)
    axs[1].grid(which='major', alpha=0.7)
    axs[1].set_xlim(left=0)
    axs[1].set_xlim(right=n - 1)
    plt.show()
    

def plot_acc_loss_aux(train_accuracies, train_digit_accuracies, train_losses, test_accuracies, test_digit_accuracies, test_losses):
    """
    Plots the training results with additional data coming from auxiliary outputs
    """
    n = len(train_accuracies)
    major_ticks = list(range(0, n, 10))
    minor_ticks = list(range(0, n, 1))

    fig, axs = plt.subplots(2, dpi=240, figsize=(15, 12))
    axs[0].plot(train_accuracies, color='Blue')
    axs[0].plot(test_accuracies, color='Red')
    axs[0].plot(train_digit_accuracies, color='Green')
    axs[0].plot(test_digit_accuracies, color='Orange')
    axs[0].set_title("Accuracy")
    axs[0].set(xlabel='Training epochs', ylabel='Accuracy (%)')
    axs[0].grid()
    axs[0].legend(['Train set final acc', 'Test set final acc', 'Train set digit acc', 'Test set digit acc'])
    axs[0].set_xlim(left=0)
    axs[0].set_xlim(right=n - 1)
    axs[0].set_xticks(major_ticks)
    axs[0].set_xticks(minor_ticks, minor=True)
    axs[0].grid(which='minor', alpha=0.3)
    axs[0].grid(which='major', alpha=0.7)

    axs[1].plot(train_losses, color='Blue')
    axs[1].plot(test_losses, color='Red')
    axs[1].set_title("Cross-Entropy Loss")
    axs[1].set(xlabel='Training epochs', ylabel='Loss')
    axs[1].grid()
    axs[1].legend(['Train set', 'Test set'])
    axs[1].set_xticks(major_ticks)
    axs[1].set_xticks(minor_ticks, minor=True)
    axs[1].grid(which='minor', alpha=0.3)
    axs[1].grid(which='major', alpha=0.7)
    axs[1].set_xlim(left=0)
    axs[1].set_xlim(right=n - 1)
    plt.show()
