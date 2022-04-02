import numpy as np
import torch
import torch.nn as nn
import warnings
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd

if not torch.cuda.is_available():
    warnings.warn('CUDA is not available.')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DNA_ONE_HOT = {'A': [0, 0, 0, 1],
               'C': [0, 0, 1, 0],
               'G': [0, 1, 0, 0],
               'T': [1, 0, 0, 0]
               }
MAX_LEN = 60

class MLP(nn.Module) :
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_size, 512), nn.LeakyReLU(), nn.Dropout(0.25), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.LeakyReLU(), nn.Dropout(0.25), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.LeakyReLU(), nn.Dropout(0.25), nn.LayerNorm(512),
            nn.Linear(512, 512), nn.LeakyReLU(), nn.Dropout(0.25), nn.LayerNorm(512),
            nn.Linear(512, self.output_size)
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, x):
        self.eval()
        return self.forward(x)


def train_epoch(model, data_loader, optimizer: torch.optim, loss_func=nn.MSELoss(), device=DEVICE):

    model.train()
    train_loss = 0
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        x = batch[0]
        y = batch[1]
        x = x.to(device)
        y = y.to(device)

        output = model.forward(x)
        loss = loss_func(output.squeeze(), y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_loss += loss.cpu().numpy()

    mean_loss = train_loss/len(data_loader)
    print("Train loss :", mean_loss)

    return mean_loss


def valid_epoch(model, data_loader, loss_func=nn.MSELoss(), device=DEVICE):

    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            x = batch[0]
            y = batch[1]
            x = x.to(device)
            y = y.to(device)

            output = model.forward(x)
            loss = loss_func(output.squeeze(), y)
            valid_loss += loss.sum().cpu().numpy()

    mean_loss = valid_loss/len(data_loader)
    print("Validation loss :", mean_loss)

    return mean_loss


def train(model, train_dataset, valid_dataset, epochs, learning_rate=0.1, batch_size=1, loss_func=nn.MSELoss(), device=DEVICE):

    model.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    for i in range(epochs):
        print('epoch : ', i+1)
        train_loss = train_epoch(model, train_dataloader, optimizer, loss_func, device)
        valid_loss = valid_epoch(model, valid_dataloader, loss_func, device)

    return train_loss, valid_loss

def make_dataset(file_name='variable_length_dataset.csv', targets=DNA_ONE_HOT):
    dataset = pd.read_csv(file_name, header=None, delimiter=',', dtype={0: str, 1: float})
    # np_data = dataset.to_numpy()
    x_array = dataset.values[:, 0].astype(str)
    y_array = dataset.values[:, 1].astype(float)

    # *** with max length of aptamer sequence 60 ***
    x_one_hot_array = np.zeros((len(x_array), MAX_LEN, 4))

    for i, sequence in enumerate(x_array):
        one_hot = np.array([targets[letter] for letter in sequence])
        padded_one_hot = np.pad(one_hot, ((0, MAX_LEN-one_hot.shape[0]), (0, 0)), mode='constant', constant_values=0)
        x_one_hot_array[i] = padded_one_hot

    x_tensor = torch.tensor(x_one_hot_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_array, dtype=torch.float32)

    return TensorDataset(x_tensor, y_tensor)


# ***
# split dataset in 2 with a proportion
# Usage :
# -use once for train,valid
# -use twice for train,valid, test (recall function on second split)
# ***
def split_dataset(dataset, seed=42, split=0.8):
    n_examples = len(dataset)
    nb_train = int(n_examples * split)
    train, test = random_split(dataset, [nb_train, n_examples-nb_train], generator=torch.Generator().manual_seed(seed))

    return train, test


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        return torch.sqrt(self.mse(pred, y))


def get_test_loss(dataloader, model, loss_func=RMSELoss()):

    for batch in dataloader:
        x = batch[0]
        y = batch[1]
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        pred = model.predict(x)
        loss = loss_func
        loss_value = loss(pred.squeeze(), y).item()
        print('Test loss : ', loss_value)

        return loss_value


# dataset = make_dataset('1mill_dataset.csv')
# train_set, validation_set = split_dataset(dataset)
# validation_set, test_set = split_dataset(validation_set, split=0.9)
#
# test_dataloader = DataLoader(test_set, batch_size=len(test_set))
#
# lrs = [0.005, 0.001, 0.0005, 0.0001]
# epochs = [10, 25, 50, 100, 150]
# batch_sizes = [1, 5, 10, 50, 100, 1000, 5000, 10000]
# num_experiments = len(lrs) * len(epochs) * len(batch_sizes)
# experiments = np.empty((num_experiments, 6))
#
# experiment_index = 0

# for lr in lrs:
#     for epoch in epochs:
#         for batch_size in batch_sizes:
#             if experiment_index >= 59:
#                 mlp_model = MLP(240, 1)
#                 train_loss, valid_loss = train(mlp_model, train_set, validation_set, epochs=50, learning_rate=0.001, batch_size=5000, loss_func=RMSELoss(), device=DEVICE)
#                 test_loss = get_test_loss(test_dataloader, mlp_model)
#                 experiments[experiment_index] = np.array([lr, epoch, batch_size, train_loss, valid_loss, test_loss])

            # experiment_index += 1


# np.save('exp_results.npy', experiments)

# torch.save(mlp_model.state_dict(), './mlp_model_1_mill.pt')
#
# mlp_model.load_state_dict(torch.load('./mlp_model_1_mill.pt'))
# mlp_model.eval()
# mlp_model.to(DEVICE)

res = np.load('exp_results.npy')

print('done')

