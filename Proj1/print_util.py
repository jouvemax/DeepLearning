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
    print('\r' + color + title.ljust(14) +
          '[{0}/{1}]'.format(b+1, batch_length).ljust(14) +
          '{batch_time.sum:.0f}s'.format(batch_time=batch_time).ljust(7) +
          '{loss.val:.4f} ({loss.avg:.4f})'.format(loss=loss).ljust(25) +
          '{top1.val:.3f} ({top1.avg:.3f})'.format(top1=top1).ljust(22) + Color.END,
          end='\n' if persistent else '')


def log_acc_loss_header(color=''):
    print(color + 'Epoch'.ljust(12) + 'Time'.ljust(8) + 'Train loss'.ljust(15) +
          'Train accuracy'.ljust(20) + 'Test loss'.ljust(15) + 'Test accuracy'.ljust(20) + Color.END)


def log_acc_loss(e, nb_epoch, time, train_loss, train_acc, test_loss, test_acc, color='', persistent=True):
    print('\r' + color +
          '[{0}/{1}]'.format(e + 1, nb_epoch).ljust(12) +
          '{time:.0f}s'.format(time=time).ljust(8) +
          '{0:.4f}'.format(train_loss).ljust(15) +
          '{0:.4f}'.format(train_acc).ljust(20) +
          '{0:.4f}'.format(test_loss).ljust(15) +
          '{0:.4f}'.format(test_acc).ljust(20) +
          Color.END,
          end='\n' if persistent else '')

