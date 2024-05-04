import xml.etree.ElementTree as ET
import svg.path
import numpy as np
import cv2
import itertools
from PIL import ImageOps, Image
from tslearn.metrics import dtw
import os

from Exercise_4.main import transfrom_rank_into_word

def read_keywords():
    keywords_file = "Exercise_4/KWS-test/keywords.tsv"
    with open(keywords_file, "r") as f:
        lines = f.readlines()
        keywords = [line.split("\t")[0] for line in lines]
    return keywords

def read_jpg_file(filenumber):
    jpg_file = f"Exercise_4/KWS-test/images/{filenumber}.jpg"
    with open(jpg_file, 'rb') as f:  # Open in binary mode ('rb') instead of text mode ('r')
        jpg_data = f.read()
    root = ET.fromstring(jpg_data)
    words = []
    ids = []
    for path in root.findall('.//{http://www.w3.org/2000/svg}path'):
        commands = path.attrib['d']
        id = path.attrib['id']
        words.append(commands)
        ids.append(id)
    return words, ids

def get_images_from_words(filenumber, words):
    img_file = f"Exercise_4/KWS-test/images/{filenumber}.jpg"
    img = Image.open(img_file)
    word_polygons = []
    word_images = []
    word_id = []
    for i, word in enumerate(words):
        path = svg.path.parse_path(word)
        xmin, ymin, xmax, ymax = get_path_box(path=path)
        word_img = img.crop((xmin, ymin, xmax, ymax))
        word_arr = np.array(word_img)
        word_polygons.append(path)
        word_images.append(word_arr)
    return word_polygons, word_images

def get_binarized_images(word_images):
    binarized_word_images = []
    for word_img in word_images:
        target_size = (120, 90)
        word_img = cv2.resize(word_img, target_size)
        word_img_gray = ImageOps.grayscale(Image.fromarray(word_img))
        threshold = 128
        word_img_bw = ImageOps.invert(word_img_gray).point(lambda x: 0 if x < threshold else 255, '1')
        binarized_word_images.append(np.array(word_img_bw))
    return binarized_word_images

def sliding_window(input_image, window_length, window_off_set):
    pos1 = 0
    pos2 = window_length
    out_windows = []
    while(pos2 < input_image.shape[1]):
        temp_window = input_image[:, pos1 : pos2]
        feature_window = []
        upper_conture = upper_conture_location(temp_window)
        feature_window.append(upper_conture)
        lower_conture = lower_conture_location(temp_window)
        feature_window.append(lower_conture)
        feature_window.append(fraction_of_black_pixels(temp_window))
        feature_window.append(fraction_of_black_pixels_between_contures(temp_window, lower_conture, upper_conture))
        feature_window.append(num_of_transitions(temp_window, lower_conture, upper_conture))
        pos1 = (pos2 + window_off_set)
        pos2 += (window_length + window_off_set)
        out_windows.append(feature_window)
    return np.array(out_windows)

def fraction_of_black_pixels(window):
    temp_window = window.flatten()
    return temp_window.sum()/temp_window.shape[0]

def upper_conture_location(window):
    pos = window.shape[0] - 1
    while(pos > 0):
        if(window[pos].any()):
            break
        pos -= 1
    return pos

def lower_conture_location(window):
    pos = 0
    while(pos < window.shape[0]):
        if(window[pos].any()):
            break
        pos += 1
    return pos

def fraction_of_black_pixels_between_contures(window, lower_conture, upper_conture):
    if lower_conture > upper_conture:
        return 0.0
    elif lower_conture == upper_conture:
        return 1 / window.shape[0]*window.shape[1]
    else:
        return fraction_of_black_pixels(window[lower_conture : upper_conture])

def num_of_transitions(window, lower_conture, upper_conture):
    if (lower_conture > upper_conture):
        return 0
    else:
        y_axis = window.shape[1] // 2
        pos = 0
        transistions = 0
        last_point = window[pos, y_axis]
        while(pos < window.shape[0]):
            cur_point = window[pos, y_axis]
            if(last_point != cur_point):
                transistions += 1
            last_point = cur_point
            pos += 1
        return transistions

def feature_matrices(binarized_word_images):
    feature_matrices = []
    for pic in binarized_word_images:
        feature_matrices.append(sliding_window(pic, 1, 1))
    return feature_matrices

def find_dtw(validation_set, train_set):
    dtw_matrix = np.zeros(shape = (len(validation_set), len(train_set)))
    for i in range(0 , len(validation_set)):
        for j in range(0, len(train_set)):
            dtw_matrix[i, j] = dtw(validation_set[i], train_set[j], global_constraint="sakoe_chiba")
    return dtw_matrix

def rank_dtw_distances(dtw_distances):
    ranked_dtw_distances = np.argsort(dtw_distances, axis = 1)
    return ranked_dtw_distances

def calculate_precision_and_recall(precision_top_ranks, keywords, ranked_train_words, validation_transcription):
    precisions = [1]
    recall = []

    for precision in precision_top_ranks:
        true_positive = 0
        false_positive = 0
        false_negative = 0
        keywords_length = len(keywords)
        for keyword in keywords:
            if keyword in validation_transcription:
                index = validation_transcription.index(keyword)
                top_precision_words = ranked_train_words[index][:precision]
                if (keyword in top_precision_words):
                    true_positive += 1
                    false_positive += precision - 1
                else:
                    false_positive += precision
                    false_negative += 1
            else:
                keywords_length -= 1

        if precision == precision_top_ranks[0]:
            print("Keywords actually found in validation set: " + str(keywords_length))
            print("Number of total Keywords: " + str(len(keywords)))

        precisions.append(true_positive/(true_positive+false_positive))
        recall.append(true_positive/(true_positive+false_negative))

    recall.append(1)
    return precisions, recall

def draw_precision_recall_curve(precision, recall):
    import matplotlib.pyplot as plt
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

def main():
    keywords = read_keywords()
    test_set = [305, 306, 307, 308, 309]

    train_set = []
    train_transcription = []

    for i in range(305, 309):
        words, ids = read_jpg_file(i)
        word_polygons, word_images = get_images_from_words(i, words)
        binarized_word_images = get_binarized_images(word_images)
        train_set += feature_matrices(binarized_word_images)
        train_transcription += ids

    validation_set = []

    for i in test_set:
        words, ids = read_jpg_file(i)
        word_polygons, word_images = get_images_from_words(i, words)
        binarized_word_images = get_binarized_images(word_images)
        validation_set += feature_matrices(binarized_word_images)

    dtw_distances = find_dtw(validation_set, train_set)
    ranked_dtw_distances = rank_dtw_distances(dtw_distances)

    ranked_train_words = transfrom_rank_into_word(ranked_dtw_distances, train_transcription)
    validation_transcription = [item for sublist in [read_jpg_file(i)[1] for i in test_set] for item in sublist]

    precision_top_ranks = np.arange(1, len(train_transcription))
    precision, recall = calculate_precision_and_recall(precision_top_ranks, keywords, ranked_train_words, validation_transcription)

    draw_precision_recall_curve(precision, recall)

if __name__ == "__main__":
    main()
