import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

map = 'brc300d'
no_landmarks = 8

def get_points(input_csv_file):
    df = pandas.read_csv('Coordinates/' + input_csv_file + '.csv', header=None)
    df_np = df.values
    return df_np


def get_actual_distance_matrix(input_csv_file):
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    df_np = df.values
    return df_np


def get_predicted_distance_matrix_landmark(input_csv_file):
    df_land = pandas.read_csv('Landmark_makeshift/' + input_csv_file + '_'+str(no_landmarks)+'.csv', header=None)
    landmark_mat = df_land.values

    df_land_ids = pandas.read_csv('Landmark_makeshift_list/' + input_csv_file + '_'+str(no_landmarks)+'.csv', header=None)
    land_ids = df_land_ids.values
    return land_ids, landmark_mat


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


coordinates = get_points(map)
actual_distance_matrix = get_actual_distance_matrix(map)
region_array, region_dict = get_region_matrix(map)

landmark_ids, predicted_distance_matrix_landmark = get_predicted_distance_matrix_landmark(map)
SE_matrix_landmark = (actual_distance_matrix - predicted_distance_matrix_landmark) ** 2
print("MSE Landmark: {}".format(np.mean(SE_matrix_landmark)))

landmark_coordinates = coordinates[landmark_ids][0]

predicted_distance_matrix_euclidean = get_predicted_distance_matrix_euclidean(map)
SE_matrix_euclidean = (actual_distance_matrix - predicted_distance_matrix_euclidean) ** 2
print("MSE Euclidean: {}".format(np.mean(SE_matrix_euclidean)))

######################

# predicted_distance_matrix_dnn_grid = get_predicted_distance_matrix_DNN_grid(map)
# SE_matrix_dnn_grid = (actual_distance_matrix - predicted_distance_matrix_dnn_grid) ** 2
# print("MSE DNN grid: {}".format(np.mean(SE_matrix_dnn_grid)))
#
# predicted_distance_matrix_dnn_K = get_predicted_distance_matrix_DNN_K(map)
# SE_matrix_dnn_K = (actual_distance_matrix - predicted_distance_matrix_dnn_K) ** 2
# print("MSE DNN K: {}".format(np.mean(SE_matrix_dnn_K)))

#######################


def get_nodewise_sq_error(SE_matrix):
    num_nodes = len(SE_matrix)
    sum_error = np.sum(SE_matrix, axis=1)
    sum_error /= num_nodes
    return sum_error


def draw_heatmap(point_coordinates, error_mat, figure, axis, caption):
    im = axis.scatter(point_coordinates[:, 0], point_coordinates[:, 1], c=error_mat, cmap='viridis')
    figure.colorbar(im, ax=axis)
    axis.title.set_text(caption)

    return

def get_dist_to_max_SE(SE_matrix):

    sum_error = get_nodewise_sq_error(SE_matrix)
    max_err_node_id = np.argmax(sum_error)

    return max_err_node_id, SE_matrix[max_err_node_id]


# sum_error_euclidean = get_nodewise_sq_error(SE_matrix_euclidean)
# plt.plot(np.arange(sum_error_euclidean.shape[0]), sorted(sum_error_euclidean), color='red', label='euclidean')
#
# sum_error_landmark = get_nodewise_sq_error(SE_matrix_landmark)
# plt.plot(np.arange(sum_error_landmark.shape[0]), sorted(sum_error_landmark), color='blue', label='landmark')
#
# sum_error_dnn_K = get_nodewise_sq_error(SE_matrix_dnn_K)
# plt.plot(np.arange(sum_error_dnn_K.shape[0]), sorted(sum_error_dnn_K), color='green', label='dnn_K')
#
# sum_error_dnn_grid = get_nodewise_sq_error(SE_matrix_dnn_grid)
# plt.plot(np.arange(sum_error_dnn_K.shape[0]), sorted(sum_error_dnn_grid), color='purple', label='dnn_grid')
#
# plt.legend()
# plt.show()
#
# sum_error_euclidean = get_nodewise_sq_error(SE_matrix_euclidean)
# plt.plot(np.arange(sum_error_euclidean.shape[0]), sum_error_euclidean, color='red', label='euclidean')
#
# sum_error_landmark = get_nodewise_sq_error(SE_matrix_landmark)
# plt.plot(np.arange(sum_error_landmark.shape[0]), sum_error_landmark, color='blue', label='landmark')
#
# sum_error_dnn_K = get_nodewise_sq_error(SE_matrix_dnn_K)
# plt.plot(np.arange(sum_error_dnn_K.shape[0]), sum_error_dnn_K, color='green', label='dnn_K')
#
# sum_error_dnn_grid = get_nodewise_sq_error(SE_matrix_dnn_grid)
# plt.plot(np.arange(sum_error_dnn_K.shape[0]), sum_error_dnn_grid, color='purple', label='dnn_grid')
#
# plt.legend()
# plt.show()


fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 4))

r1, r2 = axes
c1, c2 = r1
c3, c4 = r2

img = plt.imread('Images/' + map + '.png')
c1.imshow(img)
c1.title.set_text('Original Image')

draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_euclidean), fig, c2, 'euclidean: ' + str(np.mean(SE_matrix_euclidean)))
draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_landmark), fig, c3, 'landmark: ' + str(np.mean(SE_matrix_landmark)))
c3.scatter(landmark_coordinates[:, 0], landmark_coordinates[:, 1], c='red', cmap='viridis')

plt.show()
#draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_dnn_K), fig, c4, 'dnn_K: ' + str(np.mean(SE_matrix_dnn_K)))

fig_2, axes_2 = plt.subplots(ncols=2, nrows=2, figsize=(8, 4))

r1_2, r2_2 = axes_2
c1_2, c2_2 = r1_2
c3_2, c4_2 = r2_2

draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_euclidean), fig_2, c1_2, 'euclidean: ' + str(np.mean(SE_matrix_euclidean)))

max_err_node_id_euclidean, SE_max_euclidean =  get_dist_to_max_SE(SE_matrix_euclidean)
draw_heatmap(coordinates, SE_max_euclidean, fig_2, c2_2, 'euclidean_max: ' + str(np.mean(SE_max_euclidean)))
c2_2.scatter(coordinates[max_err_node_id_euclidean][0],coordinates[max_err_node_id_euclidean][1], c='red', cmap='viridis')

draw_heatmap(coordinates, get_nodewise_sq_error(SE_matrix_landmark), fig_2, c3_2, 'landmark: ' + str(np.mean(SE_matrix_landmark)))
c3_2.scatter(landmark_coordinates[:, 0], landmark_coordinates[:, 1], c='red', cmap='viridis')

max_err_node_id_landmark, SE_max_landmark = get_dist_to_max_SE(SE_matrix_landmark)
draw_heatmap(coordinates, SE_max_landmark, fig_2, c4_2, 'landmark_max: ' + str(np.mean(SE_max_landmark)))
c4_2.scatter(coordinates[max_err_node_id_landmark][0],coordinates[max_err_node_id_landmark][1], c='red', cmap='viridis')




plt.show()


