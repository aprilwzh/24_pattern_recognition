import csv
import numpy as np
import imageio.v3 as iio


def read_file(folder_name, file_name, train=True):
    n_files = -1
    if train:
        n_files = 60000
    else:
        n_files = 10000

    file_names = [None] * n_files
    labels = np.zeros(n_files, dtype='uint8') - 1
    with open(folder_name + file_name) as file:
        tsv_file = csv.reader(file, delimiter='\t')

        for i, line in enumerate(tsv_file):
            file_names[i] = line[0]
            labels[i] = line[1]

    return file_names, labels


def load_files(folder_name, file_names):
    files = np.zeros((len(file_names), 28, 28), dtype='int16')
    print(file_names[10])
    for i, file in enumerate(file_names):
        try:
            files[i] = iio.imread(folder_name + file)
        except TypeError:
            print(file)

    return files
