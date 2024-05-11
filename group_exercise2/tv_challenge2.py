import numpy as np

from utils import *
import os
import csv

keywords_locations = pandas.read_csv('KWS-test/keywords.tsv', sep='\t', header=None)

keywords = keywords_locations[0].values
locations = keywords_locations[1].values


def get_train_images(locations):
    lines = []
    for location in locations:
        lines.append(location.split('-'))

    files = []
    for line in lines:
        files.append(int(line[0]))

    return lines, np.array(files).astype(np.int64)


lines, files = get_train_images(locations)

file_idxs = np.unique(files)
train_files = file_idxs[file_idxs < 300]
val_files = file_idxs[file_idxs >= 300]
train_locations = locations[files < 300]
val_locations = locations[files >= 300]

window_width = 1
offset = 1

train_set, train_file_indices = get_feature_matrices(train_files, window_width=1, offset=1,
                                                     specific_imgs=train_locations)
val_set, val_file_indices = get_feature_matrices(val_files, window_width=1, offset=1, specific_imgs=val_locations)

combined_set = train_set + val_set
combined_file_indices = train_file_indices + val_file_indices
print(len(combined_set))

test_files = get_file_indices('KWS-test/test.tsv')

test_set, test_file_indices = get_feature_matrices(test_files, window_width=window_width, offset=offset,
                                                   folder_name='KWS-test/')

# print(test_file_indices)

print(len(combined_set), len(test_set))
dtw_distance = find_dtw(test_set, combined_set)
sorted_dtw_distances = np.sort(dtw_distance, axis=1)
np.savetxt('distances2.txt', sorted_dtw_distances)
print('finished part 1')

sorted_idxs = np.argsort(dtw_distance, axis=1)
sorted_dtw_distances = dtw_distance[np.arange(len(dtw_distance))[:, None], sorted_idxs]
test_file_indices = np.array(test_file_indices)
sorted_test_indices = test_file_indices[sorted_idxs]


def save_results_to_tsv(sorted_dtw_distances, keywords, sorted_test_indices, output_path='test.tsv'):
    with open(output_path, 'w') as f:
        for i in range(len(keywords)):
            keyword = keywords[i]
            distances = sorted_dtw_distances[i]
            indices = sorted_test_indices[i]
            f.write(f'{keyword} ')
            for j in range(len(distances)):
                index = indices[j]
                distance = distances[j]
                f.write(f'{index}\t{distance} ')
            f.write(f'\n')

sorted_dtw_distances = np.loadtxt('distances2.txt')

save_results_to_tsv(sorted_dtw_distances, keywords, sorted_test_indices, output_path='test.tsv')

print('created the file')
