import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from operator import add, sub, mul, truediv
from math import inf
from pathlib import Path


img_to_tensor = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.Grayscale(),
        transforms.ToTensor()
    ]
)

img_to_tensor_pre_proc = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.Grayscale(), transforms.ToTensor()]
)


def load_classes(filename='classes.json'):
    base_path = Path(__file__)
    filepath = base_path.parent / filename
    with open(filepath, 'r') as fp:
        classes_dict = json.loads(fp.read())
    return classes_dict


def build_classes():
    classes = set()
    for n1 in range(0, 10):
        for n2 in range(0, 10):
            for operation in [add, sub, mul, truediv]:
                try:
                    classes.add(round(operation(n1, n2), 2))
                except ZeroDivisionError:
                    classes.add(inf)
    classes_dict = {f'{val:.2f}': klass for klass, val in enumerate(sorted(list(classes)))}
    classes_dict['-inf'] = len(classes_dict.keys())
    return classes_dict


def hot_encoding(operator):
    if operator in [0, '+', add]:
        return add, torch.tensor([32*[8*[1, 0, 0, 0]]])
    if operator in [1, '-', sub]:
        return sub, torch.tensor([32*[8*[0, 1, 0, 0]]])
    if operator in [2, '*', mul]:
        return mul, torch.tensor([32*[8*[0, 0, 1, 0]]])
    if operator in [3, '/', truediv]:
        return truediv, torch.tensor([32*[8*[0, 0, 0, 1]]])
    return None, torch.tensor([32*[8*[0, 0, 0, 0]]])


def adjust_inputs(images_a, images_b, operators):
    fixed_input = torch.empty(size=(len(operators), 3, 32, 32))
    for i in range(len(operators)):
        tensor = torch.cat(
            (img_to_tensor_pre_proc(images_a[i]), img_to_tensor_pre_proc(images_b[i]), hot_encoding(operators[i])[1])
        )
        fixed_input[i] = tensor
    return fixed_input


def get_accuracy(model, data_loader, device):
    correct_pred = 0
    n = 0
    with torch.no_grad():
        model.eval()
        X, y_true = data_loader
        for i in range(len(y_true)):
            X_i = X[i]
            y_true_i = y_true[i]
            X_i = X_i.to(device)
            y_true_i = y_true_i.to(device)
            y_prob = model(X_i)
            _, predicted_labels = torch.max(y_prob, 1)
            n += y_true_i.size(0)
            correct_pred += (predicted_labels == y_true_i).sum()
    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses):
    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs",
           xlabel='Epoch',
           ylabel='Loss')
    ax.legend()
    fig.show()

    # change the plot style to default
    plt.style.use('default')


if __name__ == '__main__':
    print(build_classes())
