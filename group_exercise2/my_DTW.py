from utils import *

train_files = get_file_indices('KWS/train.tsv')
val_files = get_file_indices('KWS/validation.tsv')

window_width = 1
offset = 1

train_set, train_file_indices = get_feature_matrices(train_files, window_width=window_width, offset=offset)
val_set, val_file_indices = get_feature_matrices(val_files, window_width=window_width, offset=offset)

print(len(train_set), len(val_set))
dtw_distance = find_dtw(train_set, val_set)
sorted_dtw_distances = np.argsort(dtw_distance, axis=1)
print('finished part 1')

# np.savetxt('distances.txt', sorted_dtw_distances)
#
# sorted_dtw_distances = np.loadtxt('distances.txt')
# # In[ ]:
# print(sorted_dtw_distances[0])

train_words, val_words = read_words(train_files)
sorted_train_words = convert_rank_to_word(sorted_dtw_distances, train_words)
keywords = read_keywords()

precision_top_1 = np.arange(1, len(train_words))
precision, recall = precision_and_recall(precision_top_1, keywords, sorted_train_words, val_words)
print(precision, recall)

plot_pres_recall(precision, recall)