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


def plot_acc_loss(train_accuracies, train_losses, test_accuracies, test_losses):
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


