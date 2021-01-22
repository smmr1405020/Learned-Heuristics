import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data_generator_2 import load_dataset, dataset_loader
import random

np.random.seed(12345)
torch.manual_seed(12345)
random.seed(12345)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map = 'brc300d'


def f_fn(delta, x):
    if torch.abs(x) <= delta:
        return delta * torch.abs(x)
    else:
        return 0.5 * (x * x + delta * delta)


class leaf_dnn(nn.Module):

    def __init__(self, total_num_nodes, embedding_dim):
        super(leaf_dnn, self).__init__()
        self.embedding_dim = embedding_dim
        self.node_embeddings = nn.Embedding(total_num_nodes, embedding_dim)

        self.fc1 = nn.Linear(2 * embedding_dim + 4, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 1)

    def forward(self, x_inp):
        embed = self.node_embeddings(x_inp[:, :2])
        embed_r = torch.reshape(embed, (x_inp.shape[0], -1))
        embed_r_c = torch.cat([embed_r, x_inp[:, 2:]], dim=-1)

        out_1 = torch.relu(self.fc1(embed_r_c))
        out_2 = self.fc2(out_1)

        return out_2


def train_model(save_path, epochs, input_csv_file):
    training_data, num_nodes, max_val, mean, std, landmark_mat = load_dataset(input_csv_file)
    print(num_nodes)

    model = leaf_dnn(num_nodes, 8)
    model.to(device)

    ds = dataset_loader(training_data)

    train_loader = DataLoader(ds, batch_size=8, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    min_train_loss = 1000000
    train_batch_num = 0

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()

        for batch_idx, (x, y_t) in enumerate(train_loader):
            x = torch.LongTensor(x).to(device)
            y_t = torch.FloatTensor(y_t).to(device)

            train_batch_num = batch_idx
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = torch.squeeze(y_pred)

            # errors = torch.abs(y_t - y_pred)
            # errors_sorted = torch.sort(errors)[0]
            # delta = errors_sorted[-6]
            #
            # mask = errors > delta
            # top = mask * errors
            # bottom = (~mask) * errors
            # new_err = 0.5 * (top * top + delta * delta) + delta * bottom
            # loss = torch.mean(new_err)

            loss = torch.nn.MSELoss().to(device)(y_t, y_pred)

            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= (train_batch_num + 1)

        if epoch % 1 == 0:
            print("epoch {}; T loss={:.4f}".
                  format(epoch, train_loss))

        if epoch == 0 or min_train_loss > train_loss:
            min_train_loss = train_loss
            torch.save(model, 'temp_model')

    print("Done")

    model = torch.load('temp_model')
    test_loader = DataLoader(ds, batch_size=num_nodes * num_nodes, shuffle=False)

    adj_mat = np.zeros((num_nodes,num_nodes))
    for batch_idx, (x, y_t) in enumerate(test_loader):
        x = torch.LongTensor(x).to(device)
        y_pred = model(x)

        adj_mat = torch.reshape(y_pred, shape=(num_nodes, -1))
        adj_mat = adj_mat.cpu().detach().numpy()

    np.savetxt(save_path, adj_mat, delimiter=",", fmt='%.3f')


train_model('Outputs/'+map+'.csv', 5, map)
