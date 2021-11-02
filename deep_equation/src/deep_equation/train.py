import torch
import datetime as dt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from deep_equation.src.deep_equation.helper import get_accuracy, plot_losses
from deep_equation.src.deep_equation.model import LeNet


class TrainAndValidate:
    def __init__(self, model=LeNet):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model().to(self.device)
        self.learning_rate = -1
        self.num_epochs = -1
        self.dataset_dir = None
        self.loss = None
        self.optimizer = None

    def setup(self,
              dataset_dir=Path("../..") / "dataset",
              learning_rate=0.01, num_epochs=10,
              loss=nn.CrossEntropyLoss(),
              optimizer=optim.Adam
              ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dataset_dir = dataset_dir
        self.loss = loss
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

    def train(self, loader):
        self.model.train()
        running_loss = 0
        for X, y_true in loader:
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y_true = y_true.to(self.device)
            # Forward pass
            y_hat = self.model(X)
            loss = self.loss(y_hat, y_true)
            running_loss += loss.item() * X.size(0)
            # Backward pass
            loss.backward()
            self.optimizer.step()
        epoch_loss = running_loss / len(loader.dataset)
        return self.model, self.optimizer, epoch_loss

    def validate(self, loader):
        self.model.eval()
        running_loss = 0
        for X, y_true in loader:
            X = X.to(self.device)
            y_true = y_true.to(self.device)
            # Forward pass and record loss
            y_hat = self.model(X)
            loss = self.loss(y_hat, y_true)
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / len(loader.dataset)
        return self.model, epoch_loss

    def training_loop(self, train_loader, valid_loader, print_every=1):
        # set objects for storing metrics
        best_loss = 1e10
        train_losses = []
        valid_losses = []
        # Train model
        for epoch in range(0, self.num_epochs):
            # training
            model, optimizer, train_loss = self.train(train_loader)
            train_losses.append(train_loss)
            # validation
            with torch.no_grad():
                model, valid_loss = self.validate(valid_loader)
                valid_losses.append(valid_loss)
            if epoch % print_every == (print_every - 1):
                train_acc = get_accuracy(model, train_loader, device=self.device)
                valid_acc = get_accuracy(model, valid_loader, device=self.device)

                print(
                    f'{dt.datetime.now().time().replace(microsecond=0)} --- '
                    f'Epoch: {epoch}\t'
                    f'Train loss: {train_loss:.4f}\t'
                    f'Valid loss: {valid_loss:.4f}\t'
                    f'Train accuracy: {100 * train_acc:.2f}\t'
                    f'Valid accuracy: {100 * valid_acc:.2f}'
                )

        plot_losses(train_losses, valid_losses)

        return model, optimizer, (train_losses, valid_losses)

    def run(self, state_dir=Path("../..") / "model_state"):
        self.setup()
        transform = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()]
        )
        train_dataset = datasets.MNIST(
            root=self.dataset_dir,
            train=True,
            transform=transform,
            download=True
        )
        valid_dataset = datasets.MNIST(
            root=self.dataset_dir,
            train=False,
            transform=transform,
            download=True
        )
        train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)
        model, optimizer, _ = self.training_loop(train_loader, valid_loader)
        now = dt.datetime.now()
        state_filename = "{}_state.pt".format(now.strftime("%Y%m%d-%H%M%S"))
        full_path = state_dir / state_filename
        with open(full_path, 'w'):
            torch.save(model.state_dict(), full_path)


if __name__ == '__main__':
    tv = TrainAndValidate()
    tv.run()
