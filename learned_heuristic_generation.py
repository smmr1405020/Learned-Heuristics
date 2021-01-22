from model_train import train_model
import os
import sys


def generate_outputs():
    # if len(sys.argv) < 2:
    #     a = 1
    #     # input_dir = 'Dataset'
    #     # output_dir = 'Outputs'
    # else:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    input_dir = input_dir.replace(" ", "\\ ").replace("?", "\\?").replace("&", "\\&"). \
        replace("(", "\\(").replace(")", "\\)").replace("*", "\\*").replace("<", "\\<").replace(">", "\\>")

    input_csv_files = os.listdir(input_dir)

    for input_csv_file in input_csv_files:
        if input_csv_file.endswith('.csv'):
            print(input_csv_file)
            input_csv_file_path = os.path.join(input_dir, input_csv_file)
            filename = input_csv_file.split(".")
            save_path = os.path.join(output_dir, filename[0] + '_dnn_4.csv')
            train_model(save_path, 1000, input_csv_file_path)


generate_outputs()
