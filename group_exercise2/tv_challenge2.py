from utils import *
import os
import csv


keywords_locations = pandas.read_csv('group_exercise2/KWS-test/keywords.tsv', sep='\t', header=None)

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

test_files = get_file_indices('group_exercise2/KWS-test/test.tsv')

test_set, test_file_indices = get_feature_matrices(train_files, window_width=window_width, offset=offset)

print(len(combined_set), len(test_set))
dtw_distance = find_dtw(combined_set, test_set)
sorted_dtw_distances = np.argsort(dtw_distance, axis=1)
print('finished part 1')


def save_results_to_tsv(sorted_dtw_distances, keywords, output_path='group_exercise2/Challenge.tsv'):
    with open(output_path, 'w') as f:
        for i, keyword in enumerate(keywords):
            f.write(keyword)
            distances = sorted_dtw_distances[i]
            # Sorting the dissimilarities
            sorted_distances_indices = np.argsort(distances)[::-1]
            for index in sorted_distances_indices:
                f.write(f'\t{index}\t{distances[index]}')  # Write id + distance
            f.write('\n')

# Loading the distances from distances.txt
sorted_dtw_distances = np.loadtxt('distances.txt')

# save the results
save_results_to_tsv(sorted_dtw_distances, keywords)



print('created the file')
