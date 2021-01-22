import pandas
import numpy as np
from torch.utils.data import Dataset


map = 'brc300d'

def load_dataset(input_csv_file):
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    df_land = pandas.read_csv('Landmark/' + input_csv_file + '_4.csv', header=None)
    df_euclidean = pandas.read_csv('Euclidean/' + input_csv_file + '.csv', header=None)
    df_region = pandas.read_csv('Region/' + input_csv_file + '.csv', header=None)

    df_np = df.values
    landmark_mat = df_land.values
    euclidean_mat = df_euclidean.values
    region = df_region.values[0]

    max_val = np.max(df_np)
    # df_np = max_val - df_np

    mean = np.mean(df_np)
    std = np.std(df_np)

    # df_np = (df_np - mean) / std

    num_nodes = df_np.shape[0]
    all_region = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            all_region.append([region[i], region[j]])

    all_region = np.array(all_region)
    a = np.reshape(np.indices(df_np.shape), newshape=(2, -1)).T

    df_np = df_np.reshape(-1, 1)
    df_land_np = landmark_mat.reshape(-1, 1)
    df_euclidean_np = euclidean_mat.reshape(-1, 1)

    training_ds = np.concatenate([a, all_region, df_land_np, df_euclidean_np, df_np], axis=1)

    return training_ds, num_nodes, max_val, mean, std, landmark_mat


class dataset_loader(Dataset):

    def __init__(self, training_ds):
        self.x, self.y = training_ds[:, :-1], training_ds[:, -1]
        self.x = np.array(self.x).astype(np.int64)
        self.y = np.array(self.y).astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def get_adj_mat(self):
        return self.y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


load_dataset('arena')
