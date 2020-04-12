import models
import torch
import time

BATCH_SIZE = 10
LOG_INTERVAL = 20

def main():
    model = models.BaselineNetwork()
    return


def test(test_input, test_target, test_classes, model, criterion):
    with torch.no_grad():
        end = time.time()
        for b in range(0, test_input.size(0), BATCH_SIZE):
            output = model(test_input.narrow(0, b, BATCH_SIZE))
            output = torch.argmax(output)
            loss = criterion(output, test_target.narrow(0, b, BATCH_SIZE))

            errors =
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if b % LOG_INTERVAL == 0:
                print_test(batch_idx, len(val_loader), batch_time, losses, top1, persistent=False, color=color, title=title)

        print_test(len(val_loader) - 1, len(val_loader), batch_time, losses, top1, persistent=True, color=color, title=title)


if __name__ == '__main__':
    main()

