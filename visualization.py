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


def get_region_matrix(input_csv_file):
    df_region = pandas.read_csv('Region/' + input_csv_file + '.csv', header=None)
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

# predicted_distance_matrix_dnn = get_predicted_distance_matrix_DNN(map)
# SE_matrix_dnn = (actual_distance_matrix - predicted_distance_matrix_dnn) ** 2
# print("MSE DNN: {}".format(np.mean(SE_matrix_dnn)))


def get_nodewise_sq_error_sorted(SE_matrix):
    num_nodes = len(SE_matrix)
    sum_error = np.sum(SE_matrix, axis=1)
    sum_error /= num_nodes
    # print(np.sort(sum_error))
    # print(np.argsort(sum_error))
    return sum_error


def get_region_error(SE_matrix, region_dict):
    grid_size = 100

    num_regions = grid_size * grid_size
    region_error_mat = np.zeros((grid_size * grid_size))

    for i in range(num_regions):
        if i in region_dict.keys():
            nodes_i = region_dict[i]
            distances = []
            for k in range(len(nodes_i)):
                distances.append(np.mean(SE_matrix[nodes_i[k], :]))

            mean_d = np.mean(distances)
            region_error_mat[i] = mean_d

    region_error_mat = region_error_mat.reshape(grid_size, -1)

    return region_error_mat


def draw_region_heatmap(region_error_mat):
    plt.imshow(region_error_mat, cmap='viridis')
    plt.colorbar()
    plt.show()
    return


# sum_error_landmark = get_nodewise_sq_error_sorted(SE_matrix_landmark)
# plt.plot(np.arange(sum_error_landmark.shape[0]), sorted(sum_error_landmark), color='blue', label='landmark')

# sum_error_euclidean = get_nodewise_sq_error_sorted(SE_matrix_euclidean)
# plt.plot(np.arange(sum_error_euclidean.shape[0]), sorted(sum_error_euclidean), color='red', label='euclidean')
#
# sum_error_dnn = get_nodewise_sq_error_sorted(SE_matrix_dnn)
# plt.plot(np.arange(sum_error_dnn.shape[0]), sorted(sum_error_dnn), color='green', label='dnn')

# plt.legend()
# plt.show()

region_error_mat_euclidean = get_region_error(SE_matrix_euclidean, region_dict)
draw_region_heatmap(region_error_mat_euclidean)

region_error_mat_landmark = get_region_error(SE_matrix_landmark, region_dict)
draw_region_heatmap(region_error_mat_landmark)

# region_error_mat_dnn = get_region_error(SE_matrix_dnn, region_dict)
# draw_region_heatmap(region_error_mat_euclidean)
