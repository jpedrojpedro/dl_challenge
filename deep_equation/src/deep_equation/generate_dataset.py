import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from math import inf
from random import randint
from pathlib import Path
from deep_equation.src.deep_equation.helper import hot_encoding, img_to_tensor, str_to_op, load_classes


def handle_operation(n1, n2, op):
    try:
        return round(op(float(n1), float(n2)), 2)
    except ZeroDivisionError:
        return inf


def main(loader):
    classes_dict = load_classes()
    operators = {op: hot_encoding(op) for op in str_to_op.keys()}
    X = []
    y = []
    loader_lst = list(loader)
    for batch in loader_lst:
        new_batch_inputs = torch.empty(size=(128, 3, 32, 32))
        new_batch_answers = torch.full(size=(128, ), fill_value=classes_dict['-inf'])
        images = batch[0]
        answers = batch[1]
        for idx, img in enumerate(images):
            r_idx = randint(0, len(images) - 1)
            r_idx_op = randint(0, len(operators) - 1)
            k_idx_op = list(str_to_op.keys())[r_idx_op]
            new_input = torch.cat((img, images[r_idx], operators[k_idx_op]))
            new_answer = handle_operation(answers[idx], answers[r_idx], str_to_op[k_idx_op])
            new_batch_inputs[idx] = new_input
            new_batch_answers[idx] = torch.tensor(classes_dict[str(new_answer)])
        X.append(new_batch_inputs)
        y.append(new_batch_answers)
    return X, y


if __name__ == '__main__':
    dataset_dir = Path("../..") / "dataset"
    train_dataset = datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=img_to_tensor,
        download=True
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    X, y = main(train_loader)
