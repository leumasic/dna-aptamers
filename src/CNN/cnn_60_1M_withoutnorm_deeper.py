import numpy as np
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
import pandas as pd
from sklearn.model_selection import train_test_split


if not torch.cuda.is_available():
    warnings.warn("CUDA is not available.")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DNA_ONE_HOT = {
    "A": [0, 0, 0, 1],
    "C": [0, 0, 1, 0],
    "G": [0, 1, 0, 0],
    "T": [1, 0, 0, 0],
}
MAX_LEN = 60


def make_dataset(file_name="variable_length_dataset.csv", targets=DNA_ONE_HOT):
    dataset = pd.read_csv(
        file_name, header=None, delimiter=",", dtype={0: str, 1: float}
    )
    # np_data = dataset.to_numpy()
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    x_array_train = train.values[:, 0].astype(str)
    y_array_train = train.values[:, 1].astype(float)
    #mean = y_array_train.mean()
    #std = y_array_train.std()
    #y_array_train_norm = (y_array_train - mean) / std

    x_array_test = test.values[:, 0].astype(str)
    y_array_test = test.values[:, 1].astype(float)
    #y_array_test_norm = (y_array_test - mean) / std

    # *** with max length of aptamer sequence 60 ***
    x_one_hot_array_train = np.zeros((len(x_array_train), MAX_LEN, 4))

    for i, sequence in enumerate(x_array_train):
        one_hot = np.array([targets[letter] for letter in sequence])
        padded_one_hot = np.pad(
            one_hot,
            ((0, MAX_LEN - one_hot.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        x_one_hot_array_train[i] = padded_one_hot

    x_tensor_train = torch.tensor(x_one_hot_array_train, dtype=torch.float32)
    y_tensor_train = torch.tensor(y_array_train, dtype=torch.float32)
    train_set = TensorDataset(x_tensor_train, y_tensor_train)

    x_one_hot_array_test = np.zeros((len(x_array_test), MAX_LEN, 4))

    for i, sequence in enumerate(x_array_test):
        one_hot = np.array([targets[letter] for letter in sequence])
        padded_one_hot = np.pad(
            one_hot,
            ((0, MAX_LEN - one_hot.shape[0]), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        x_one_hot_array_test[i] = padded_one_hot

    x_tensor_test = torch.tensor(x_one_hot_array_test, dtype=torch.float32)
    y_tensor_test = torch.tensor(y_array_test, dtype=torch.float32)
    test_set = TensorDataset(x_tensor_test, y_tensor_test)

    return train_set, test_set


# def split_dataset(dataset, seed=42, train_split=0.8):
#     n_examples = len(dataset)
#     nb_train = int(n_examples * train_split)
#     train, test = random_split(
#         dataset,
#         [nb_train, n_examples - nb_train],
#         generator=torch.Generator().manual_seed(seed),
#     )

#     return train, test


# def label_normaliser(train_set, validation_set):
#     # print(train_set[1])
#     train_x, train_y = train_set
#     valid_x, valid_y = validation_set
#     mean = train_y.mean()
#     std = train_y.std()
#     train_label_norm = (train_y - mean) / std
#     validation_label_norm = (valid_y - mean) / std
#     return TensorDataset(train_x, train_label_norm), TensorDataset(
#         valid_x, validation_label_norm
#     )


num_epochs = 400
batch_size = 100
learning_rate = 0.001
ksize = 3
nfilter = 128
l1 = 32


class ConvNet(nn.Module):
    def __init__(self, nfilter, ksize, l1, batch_size):
        super(ConvNet, self).__init__()
        self.batch_size = batch_size
        self.network = nn.Sequential(
            nn.Conv1d(4, nfilter, ksize),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(nfilter, nfilter, ksize),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(),
            nn.Conv1d(nfilter, nfilter, ksize),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(nfilter, nfilter, ksize),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(nfilter, nfilter, ksize),
            nn.BatchNorm1d(nfilter),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.LazyLinear(l1),
            nn.ReLU(),
            nn.Linear(l1, 1),
        )

    def forward(self, x):
        # x --> (batch_size, 60, 4)
        # x = x.view(self.batch_size, 1, 4 * MAX_LEN)
        # x --> (batch_size, 4, 60)
        x = x.view(self.batch_size, 4 , MAX_LEN)
        return self.network(x)

    def predict(self, x):
        self.eval()
        return self.forward(x)


def train_epoch(
    model, data_loader, optimizer: torch.optim, loss_func=nn.MSELoss(), device=DEVICE
):

    model.train()
    train_loss = 0
    for batch in tqdm(data_loader):
        x , y = batch
        optimizer.zero_grad()
        #x = batch[0]
        #y = batch[1]
        x = x.to(device)
        y = y.to(device)

        # TODO fix float type at source
        output = model(x)
        loss = torch.sqrt(loss_func(output.squeeze(), y))
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.cpu().numpy()
    print("Train loss :", train_loss / len(data_loader))


def valid_epoch(model, data_loader, loss_func=nn.MSELoss(), device=DEVICE):

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            x , y = batch
            #x = batch[0]
            #y = batch[1]
            x = x.to(device)
            y = y.to(device)

            # TODO fix float type at source
            output = model(x)
            loss = torch.sqrt(loss_func(output.squeeze(), y))
            valid_loss += loss.cpu().numpy()
    print("Validation loss :", valid_loss / len(data_loader))


def train():
    device = DEVICE
    model = ConvNet(nfilter, ksize, l1, batch_size).to(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_set, validation_set = make_dataset(
        "variable_length_dataset(1M).csv"
    )
    # print(train_set[0])
    # train_set, validation_set = split_dataset(dataset)
    # train_set_norm, validation_set_norm = label_normaliser(train_set, validation_set)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validation_set, batch_size=batch_size)
    loss_func = nn.MSELoss()
    for i in range(num_epochs):
        train_epoch(model, train_dataloader, optimizer, loss_func, device)
        valid_epoch(model, valid_dataloader, loss_func, device)

train()
