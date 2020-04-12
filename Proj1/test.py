import models
import torch

BATCH_SIZE = 10
LOG_INTERVAL = 20

def main():
    model = models.BaselineNetwork()
    return


def test(test_input, test_target, test_classes, model, criterion):
    with torch.no_grad():
        nb_data_errors = 0
        loss_sum = 0
        for b in range(0, test_input.size(0), BATCH_SIZE):
            output = model(test_input.narrow(0, b, BATCH_SIZE))
            loss = criterion(output, test_target.narrow(0, b, BATCH_SIZE))
            loss_sum += loss
            _, predicted_classes = torch.max(output, 1)

            for k in range(BATCH_SIZE):
                if test_target[b + k] != predicted_classes[k]:
                    nb_data_errors = nb_data_errors + 1

            if b % LOG_INTERVAL == 0:
                pass
                #print("Accuracy: " + repr())
                #print_test(batch_idx, len(val_loader), batch_time, losses, top1, persistent=False, color=color, title=title)

        accuracy = 1 - (nb_data_errors / test_input.size(0))
        print("Accuracy: " + repr(accuracy * 100) + "%" + " - Loss: " + repr(loss_sum))


if __name__ == '__main__':
    main()

