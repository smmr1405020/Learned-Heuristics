import numpy as np
import csv
import pandas

map = 'brc300d'

def get_euclidean(input_csv_file):
    coordinates = []
    f = open("Cor/" + input_csv_file + ".cor", mode='r')
    for x in f:
        x_arr = x.split(sep=" ")
        coordinates.append([float(c) for c in x_arr[1:]])

    coordinates = np.array(coordinates)
    num_nodes = len(coordinates)
    euclidean = np.zeros((len(coordinates), len(coordinates)))
    for i in range(len(euclidean)):
        for j in range(i + 1, len(euclidean)):
            euclidean[i][j] = euclidean[j][i] = np.linalg.norm(coordinates[i] - coordinates[j])

    with open("Euclidean/" + input_csv_file + ".csv", mode='w', newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in euclidean:
            csv_file_writer.writerow(row)

    return coordinates, euclidean


def get_region_cluster(input_csv_file, euclidean):
    num_nodes = len(euclidean)

    all_clusters = []
    added_to_cluster = np.zeros(num_nodes)
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    actual_dist = df.values
    for i in range(num_nodes):
        if added_to_cluster[i] == 0:
            added_to_cluster[i] = 1
            new_cluster = [i]
            for j in range(i + 1, num_nodes):
                if ((actual_dist[i][j] - euclidean[i][j]) / actual_dist[i][j]) < 0.01 and added_to_cluster[j] == 0:
                    added_to_cluster[j] = 1
                    new_cluster.append(j)
            all_clusters.append(new_cluster)

    cluster_assignment = np.zeros(num_nodes)
    for i in range(len(all_clusters)):
        for j in range(len(all_clusters[i])):
            cluster_assignment[all_clusters[i][j]] = i

    return cluster_assignment


def get_region_grid(coordinates, grid_size):
    epsilon = 0.0001

    #print(coordinates)
    num_nodes = len(coordinates)
    steps = []
    ranges = []

    for i in range(coordinates.shape[1]):
        range_min, range_max = np.min(coordinates[:, i]), np.max(coordinates[:, i])
        ranges.append([range_min, range_max])
        step = (range_max - range_min) / grid_size
        steps.append(step)

    ranges = np.array(ranges)

    # for 2D coordinate only
    cluster_assignment = np.zeros(num_nodes)
    for i in range(num_nodes):
        row = max(int(np.floor((coordinates[i][0] - ranges[0][0]) / steps[0] - epsilon)), 0)
        col = max(int(np.floor((coordinates[i][1] - ranges[1][0]) / steps[1] - epsilon)), 0)
        cluster_no = grid_size * row + col
        cluster_assignment[i] = cluster_no

    for i in range(grid_size):
        print(str(ranges[0][0] + i * steps[0]) + "-" + str(ranges[0][0] + (i + 1) * steps[0]))
        print(str(ranges[1][0] + i * steps[1]) + "-" + str(ranges[1][0] + (i + 1) * steps[1]))
        print("\n\n")

    return cluster_assignment


def write_cluster_assignment(input_csv_file, cluster_assignment):
    with open("Region/" + input_csv_file + ".csv", mode='w', newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_file_writer.writerow(cluster_assignment)


coordinates, euclidean = get_euclidean(map)
cluster_assignment_K = get_region_cluster(map, euclidean)
cluster_assignment_grid = get_region_grid(coordinates, 100)

write_cluster_assignment(map, cluster_assignment_K)

