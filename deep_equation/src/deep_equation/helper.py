import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from operator import add, sub, mul, truediv
from math import inf


img_to_tensor = transforms.Compose(
    [transforms.Resize((32, 32)), transforms.Grayscale(), transforms.ToTensor()]
)

str_to_op = {
    '+': add,
    '-': sub,
    '*': mul,
    '/': truediv,
}


def build_classes():
    classes = set()
    for n1 in range(0, 10):
        for n2 in range(0, 10):
            for operation in [add, sub, mul, truediv]:
                try:
                    classes.add(round(operation(n1, n2), 2))
                except ZeroDivisionError:
                    classes.add(inf)
    classes_dict = {val: klass for klass, val in enumerate(sorted(list(classes)))}
    classes_dict[float('-inf')] = len(classes_dict.keys())
    return classes_dict


def hot_encoding(operator):
    if operator == '+':
        return torch.tensor([32*[8*[1, 0, 0, 0]]])
    if operator == '-':
        return torch.tensor([32*[8*[0, 1, 0, 0]]])
    if operator == '*':
        return torch.tensor([32*[8*[0, 0, 1, 0]]])
    if operator == '/':
        return torch.tensor([32*[8*[0, 0, 0, 1]]])
    return torch.tensor([32*[8*[0, 0, 0, 0]]])


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
