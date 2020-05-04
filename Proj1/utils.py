from dlc_practical_prologue import generate_pair_sets
import torch

def generate_data_device(n, device='cpu'):

    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(n)
    train_input = train_input.to(device=device)
    train_target = train_target.to(device=device)
    train_classes = train_classes.to(device=device)
    test_input = test_input.to(device=device)
    test_target = test_target.to(device=device)
    test_classes = test_classes.to(device=device)

    return train_input, train_target, train_classes, test_input, test_target, test_classes

def normalize_data(tensor):

    mu, std = tensor.mean(), tensor.std()
    tmp = tensor.sub(mu).div(std)

    return tmp