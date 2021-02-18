import csv

import numpy as np
import pandas

map = 'brc300d'
no_landmarks = 12


# find landmarks

def get_actual_distance_matrix(input_csv_file):
    df = pandas.read_csv('Dataset/' + input_csv_file + '.csv', header=None)
    df_np = df.values
    return df_np


actual_distance = get_actual_distance_matrix(map)


def get_predicted_distance_matrix_landmark(input_csv_file):
    df_land = pandas.read_csv('Landmark/' + input_csv_file + '_' + str(no_landmarks) + '.csv', header=None)
    landmark_mat = df_land.values
    return landmark_mat


landmark_original = get_predicted_distance_matrix_landmark(map)


def find_landmarks(actual_distance, no_landmarks):
    random_node = 0
    max_val = 0
    max_id = 0
    landmark_list = []
    landmark_id = np.argmax(actual_distance[random_node])
    landmark_list.append(landmark_id)

    for i in range(no_landmarks-1):

        max_val = 0
        max_id = 0

        for j in range(len(actual_distance)):
            curr_d = 0
            for landmark in landmark_list:
                if landmark == j:
                    curr_d = 0
                    break
                d = actual_distance[landmark][j]
                curr_d = curr_d + d

            if curr_d > max_val:
                max_val = curr_d
                max_id = j
            elif curr_d < 0:
                max_val = curr_d
                max_id = j
                break

        landmark_list.append(max_id)

    return landmark_list


landmark_list = find_landmarks(actual_distance, no_landmarks)

num_nodes = len(actual_distance)


def calc_landmark(actual_distance_, landmark_list_, input_csv_file):
    landmark_dist = np.zeros((len(actual_distance_), len(actual_distance_)))

    for i in range(len(actual_distance_)):
        for j in range(i + 1, len(actual_distance_)):
            start = i
            end = j
            max_dist = 0
            for k in range(len(landmark_list_)):
                st_dist = actual_distance_[landmark_list_[k]][start]
                en_dist = actual_distance_[landmark_list_[k]][end]
                max_dist = max(max_dist, np.abs(st_dist - en_dist))

            landmark_dist[i][j] = landmark_dist[j][i] = max_dist

    print(np.sum((landmark_dist - landmark_original) ** 2) / num_nodes ** 2)
    print(np.sum((landmark_dist - actual_distance) ** 2) / num_nodes ** 2)
    print(np.sum((landmark_original - actual_distance) ** 2) / num_nodes ** 2)

    with open("Landmark_makeshift/" + input_csv_file + "_" + str(no_landmarks) + ".csv", mode='w', newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in landmark_dist:
            csv_file_writer.writerow(row)

    with open("Landmark_makeshift_list/" + input_csv_file + "_" + str(no_landmarks) + ".csv", mode='w', newline="") as csv_file:
        csv_file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_file_writer.writerow(landmark_list_)



    return


calc_landmark(actual_distance, landmark_list, map)
