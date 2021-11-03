import torch
import datetime as dt
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from deep_equation.src.deep_equation.helper import get_accuracy, plot_losses, img_to_tensor, img_to_tensor_pre_proc
from deep_equation.src.deep_equation.model import LeNet
from deep_equation.src.deep_equation.generate_dataset import main as gen_ds


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
              learning_rate=0.001, num_epochs=30,
              loss=nn.CrossEntropyLoss(),
              optimizer=optim.Adam
              ):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dataset_dir = dataset_dir
        self.loss = loss
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)

    def train(self, batches):
        self.model.train()
        running_loss = 0
        X_train, y_train = batches
        for i in range(len(y_train)):
            X = X_train[i]
            y_true = y_train[i]
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y_true = y_true.to(self.device)
            # Forward pass
            y_hat = self.model(X)
            loss = self.loss(y_hat, y_true.type(torch.LongTensor))
            running_loss += loss.item() * X.size(0)
            # Backward pass
            loss.backward()
            self.optimizer.step()
        epoch_loss = running_loss / len(batches)
        return self.model, self.optimizer, epoch_loss

    def validate(self, batches):
        self.model.eval()
        running_loss = 0
        X_validation, y_validation = batches
        for i in range(len(y_validation)):
            X = X_validation[i]
            y_true = y_validation[i]
            X = X.to(self.device)
            y_true = y_true.to(self.device)
            # Forward pass and record loss
            y_hat = self.model(X)
            loss = self.loss(y_hat, y_true.type(torch.LongTensor))
            running_loss += loss.item() * X.size(0)
        epoch_loss = running_loss / len(batches)
        return self.model, epoch_loss

    def training_loop(self, training_loader, validation_loader, print_every=1):
        # set objects for storing metrics
        # best_loss = 1e10
        training_losses = []
        validation_losses = []
        # Train model
        for epoch in range(0, self.num_epochs):
            # training
            model, optimizer, train_loss = self.train(training_loader)
            training_losses.append(train_loss)
            # validation
            with torch.no_grad():
                model, valid_loss = self.validate(validation_loader)
                validation_losses.append(valid_loss)
            if epoch % print_every == (print_every - 1):
                train_acc = get_accuracy(model, training_loader, device=self.device)
                valid_acc = get_accuracy(model, validation_loader, device=self.device)

                print(
                    f'{dt.datetime.now().time().replace(microsecond=0)} --- '
                    f'Epoch: {epoch}\t'
                    f'Train loss: {train_loss:.4f}\t'
                    f'Valid loss: {valid_loss:.4f}\t'
                    f'Train accuracy: {100 * train_acc:.2f}\t'
                    f'Valid accuracy: {100 * valid_acc:.2f}'
                )

        plot_losses(training_losses, validation_losses)

        return model, optimizer, (training_losses, validation_losses)

    def run(self, state_dir=Path("../..") / "model_state"):
        self.setup()
        training_dataset = datasets.MNIST(
            root=self.dataset_dir,
            train=True,
            transform=img_to_tensor,
            download=True
        )
        validation_dataset = datasets.MNIST(
            root=self.dataset_dir,
            train=False,
            transform=img_to_tensor_pre_proc,
            download=True
        )
        training_loader = DataLoader(dataset=training_dataset, batch_size=128, shuffle=True)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=128, shuffle=True)
        X_train, y_train = gen_ds(training_loader)
        X_val, y_val = gen_ds(validation_loader)
        model, optimizer, _ = self.training_loop((X_train, y_train), (X_val, y_val))
        now = dt.datetime.now()
        state_filename = "{}_state.pt".format(now.strftime("%Y%m%d-%H%M%S"))
        full_path = state_dir / state_filename
        with open(full_path, 'w'):
            torch.save(model.state_dict(), full_path)


if __name__ == '__main__':
    tv = TrainAndValidate()
    tv.run()
