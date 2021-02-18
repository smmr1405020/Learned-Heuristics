import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

map = 'brc300d'


def get_actual_distance_matrix(input_csv_file):
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    df_np = df.values
    return df_np


def get_predicted_distance_matrix_landmark(input_csv_file):
    df_land = pandas.read_csv('Landmark/' + input_csv_file + '_8.csv', header=None)
    landmark_mat = df_land.values
    return landmark_mat


def get_predicted_distance_matrix_euclidean(input_csv_file):
    df_euclidean = pandas.read_csv('Euclidean/' + input_csv_file + '.csv', header=None)
    euclidean_mat = df_euclidean.values
    return euclidean_mat


def get_predicted_distance_matrix_DNN(input_csv_file):
    df_dnn = pandas.read_csv('Outputs/' + input_csv_file + '.csv', header=None)
    dnn_mat = df_dnn.values
    return dnn_mat

def get_predicted_distance_matrix_DNN_K(input_csv_file):
    df_dnn = pandas.read_csv('Outputs_K/' + input_csv_file + '.csv', header=None)
    dnn_mat = df_dnn.values
    return dnn_mat

def get_predicted_distance_matrix_DNN_grid(input_csv_file):
    df_dnn = pandas.read_csv('Outputs_grid/' + input_csv_file + '.csv', header=None)
    dnn_mat = df_dnn.values
    return dnn_mat


def get_region_matrix(input_csv_file):
    df_region = pandas.read_csv('Region_grid/' + input_csv_file + '.csv', header=None)
    region = df_region.values[0]

    region_dict = dict()
    for i in range(len(region)):
        region_dict.setdefault(region[i], []).append(i)

    return region, region_dict


actual_distance_matrix = get_actual_distance_matrix(map)
region_array, region_dict = get_region_matrix(map)

predicted_distance_matrix_landmark = get_predicted_distance_matrix_landmark(map)
SE_matrix_landmark = (actual_distance_matrix - predicted_distance_matrix_landmark) ** 2
print("MSE Landmark: {}".format(np.mean(SE_matrix_landmark)))

predicted_distance_matrix_euclidean = get_predicted_distance_matrix_euclidean(map)
SE_matrix_euclidean = (actual_distance_matrix - predicted_distance_matrix_euclidean) ** 2
print("MSE Euclidean: {}".format(np.mean(SE_matrix_euclidean)))

predicted_distance_matrix_dnn_grid = get_predicted_distance_matrix_DNN_grid(map)
SE_matrix_dnn_grid = (actual_distance_matrix - predicted_distance_matrix_dnn_grid) ** 2
print("MSE DNN grid: {}".format(np.mean(SE_matrix_dnn_grid)))

predicted_distance_matrix_dnn_K = get_predicted_distance_matrix_DNN_K(map)
SE_matrix_dnn_K = (actual_distance_matrix - predicted_distance_matrix_dnn_K) ** 2

print(np.max(SE_matrix_dnn_K,axis=0))





