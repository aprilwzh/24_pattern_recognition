# # Analyze the data # #
# Preprocessing:
# # Binarisation? Otsu, Sauvola, Niblack, etc.
# Extract word images from full page (cropping)
# # Bounding box - Easy, but can contain parts of other words
# # Polygon as clipping mask is more precise
import itertools

import cv2
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pandas
import svg.path
import xml.etree.ElementTree as ET

from tslearn.metrics import dtw


def get_bounding_box(svgpath):
    x_min = np.infty
    y_min = np.infty
    x_max = -np.infty
    y_max = -np.infty
    for section in svgpath:
        x_low = [x_min]
        x_high = [x_max]
        y_low = [y_min]
        y_high = [y_max]
        if isinstance(section, svg.path.Line):
            x_low.extend((section.start.real, section.end.real))
            x_high.extend((section.start.real, section.end.real))
            y_low.extend((section.start.imag, section.end.imag))
            y_high.extend((section.start.imag, section.end.imag))
        elif isinstance(section, svg.path.QuadraticBezier):
            x_low.extend((section.start.real, section.control.real, section.end.real))
            x_high.extend((section.start.real, section.control.real, section.end.real))
            y_low.extend((section.start.imag, section.control.imag, section.end.imag))
            y_high.extend((section.start.imag, section.control.imag, section.end.imag))
        elif isinstance(section, svg.path.CubicBezier):
            x_low.extend((section.start.real, section.control1.real, section.control2.real, section.end.real))
            x_high.extend((section.start.real, section.control1.real, section.control2.real, section.end.real))
            y_low.extend((section.start.imag, section.control1.imag, section.control2.imag, section.end.imag))
            y_high.extend((section.start.imag, section.control1.imag, section.control2.imag, section.end.imag))

        x_min = min(x_low)
        x_max = max(x_high)
        y_min = min(y_low)
        y_max = max(y_high)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def read_svg_file(file_number):
    with open(f'KWS/locations/{file_number}.svg', 'r') as file:
        svg_data = file.read()
    root = ET.fromstring(svg_data)
    polygons = []
    indices = []
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        polygons.append(path.attrib['d'])
        indices.append(path.attrib['id'])

    return polygons, indices


def get_sub_images_from_polygons(file_number, polygons):
    img = cv2.imread(f'KWS/images/{file_number}.jpg')

    word_boxes = []
    word_images = []
    for i, polygon in enumerate(polygons):
        path = svg.path.parse_path(polygon)

        x_min, y_min, x_max, y_max = get_bounding_box(svgpath=path)
        word_image = np.array(img[y_min:y_max, x_min:x_max])
        word_boxes.append(path)
        word_images.append(word_image)
    return word_boxes, word_images


def binarize_images(images):
    binary_images = []
    for i, image in enumerate(images):
        im_shape = image.shape
        if 0 in im_shape:
            print(i, im_shape)
            continue
        image = cv2.resize(image, (100, 100))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print(image.shape, gray_image.shape)
        binary_images.append(np.array(gray_image <= 128))
        # print(binary_images[-1].shape)

    return binary_images


def proportion_black(window):
    temp_window = window.flatten()
    return temp_window.sum() / temp_window.shape[0]


def highest_point(window):
    idx = window.shape[0] - 1

    while idx > 0:
        if any(window[idx]):
            break
        idx -= 1

    return idx


def lowest_point(window):
    idx = 0
    while idx < window.shape[0]:
        if any(window[idx]):
            break
        idx += 1

    return idx


def proportion_back_between_high_low(window):
    lowest_idx = lowest_point(window)
    highest_idx = highest_point(window)

    if lowest_idx > highest_idx:
        return 0
    elif lowest_idx == highest_idx:
        return 1. / (window.shape[0] * window.shape[1])
    else:
        return proportion_black(window[lowest_idx:highest_idx])


def count_flops(window):
    # TODO: Improve this by only searching between lowest_idx - 1 and highest_idx + 1
    lowest_idx = lowest_point(window)
    highest_idx = highest_point(window)

    if lowest_idx > highest_idx:
        return 0
    else:
        y_center = window.shape[1] // 2
        idx = 1
        flops = 0
        prev_point = window[0, y_center]
        while idx < window.shape[0]:
            point = window[idx, y_center]
            if prev_point != point:
                flops += 1
            prev_point = point
            idx += 1
        return flops


def sliding_window(image, window_width, offset):
    left = 0
    right = window_width

    out_windows = []
    while right < image.shape[1]:
        window = np.copy(image[:, left:right])
        features = []
        features.extend((highest_point(window), lowest_point(window), proportion_black(window),
                         proportion_back_between_high_low(window), count_flops(window)))

        left = right + offset
        right += window_width + offset
        out_windows.append(features)
    return np.array(out_windows)


def feature_matrices(binarized_word_images, window_width, offset):
    return [sliding_window(image=image, window_width=window_width, offset=offset)
            for image in binarized_word_images]


###
# Get Feature Matrices for Train and Validation Sets

def get_feature_matrices(files, window_width=1, offset=1):
    features = []
    indices = []

    for file in files:
        words, idx = read_svg_file(file)
        polygons, images = get_sub_images_from_polygons(file, words)
        binarized_word_images = binarize_images(images)
        features.append(feature_matrices(binarized_word_images, window_width, offset))
        indices.append(idx)

    chained_features = list(itertools.chain.from_iterable(features))
    chained_indices = list(itertools.chain.from_iterable(indices))
    return chained_features, chained_indices


def find_dtw(train_set, val_set):
    n_val = len(val_set)
    n_train = len(train_set)
    dtw_mat = np.zeros((n_val, n_train))
    for i in range(n_val):
        for j in range(n_train):
            dtw_mat[i, j] = dtw(val_set[i], train_set[i], global_constraint="sakoe_chiba")

    return dtw_mat


def read_words(train_indices):
    file_dataframe = pandas.read_csv('KWS/transcription.tsv', sep='\t', header=None)

    train_words = []
    val_words = []
    # print(file_dataframe.shape)
    df_shape = file_dataframe.shape
    train_indices = list(train_indices.values)
    for line in range(len(file_dataframe)):
        line_split = file_dataframe[0][line].split('-')

        if int(line_split[0]) in train_indices:
            train_words.append(file_dataframe[1][line])
        else:
            val_words.append(file_dataframe[1][line])

    return train_words, val_words


def convert_rank_to_word(sorted_dtw_distances, train_words):
    train_word_ranks = []

    for val_idx in sorted_dtw_distances:
        word_rank = []
        for train_word_idx in val_idx:
            word_rank.append(train_words[int(train_word_idx)])

        train_word_ranks.append(word_rank)

    return train_word_ranks


def read_keywords():
    keywords = pandas.read_csv('KWS/keywords.tsv', sep='\t', header=None)
    keywords = keywords.values

    return keywords


def precision_and_recall(precision, keywords, sorted_train_words, val_words):
    precisions = [1]
    recall = []

    for pres in precision:
        tp = 0
        fp = 0
        fn = 0
        n_keywords = len(keywords)
        for keyword in keywords:
            if keyword in val_words:
                idx = val_words.index(keyword)
                best_words = sorted_train_words[idx][:pres]
                # print(best_words)
                # return
                if keyword in best_words:
                    tp += 1
                    fp += pres - 1
                else:
                    fp += pres
                    fn += 1
            else:
                n_keywords -= 1

        precisions.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))

    recall.append(1)
    return precisions, recall


def plot_pres_recall(precision, recall):
    plt.plot(recall, precision)
    plt.title("Precision and Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


def get_file_indices(file_name):
    files = pandas.read_csv(file_name, sep='\t', header=None)
    return files[0].apply(int)


# In[ ]:

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
