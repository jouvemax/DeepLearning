import matplotlib.pyplot as plt

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




def log_acc_loss_header(color=''):
    print(color + 'Epoch'.ljust(12) + 'Time'.ljust(8) + 'Train loss'.ljust(15) +
          'Train accuracy'.ljust(20) + 'Train F1 score'.ljust(15) + Color.END)

def log_acc_loss(e, nb_epoch, time, train_loss, train_acc, train_f, color='', persistent=True):
    print('\r' + color +
          '[{0}/{1}]'.format(e + 1, nb_epoch).ljust(12) +
          '{time:.0f}s'.format(time=time).ljust(8) +
          '{0:.4f}'.format(train_loss).ljust(15) +
          '{0:.4f}'.format(train_acc).ljust(20) +
          '{0:.4f}'.format(train_f).ljust(15) +
          Color.END,
          end='\n' if persistent else '')
