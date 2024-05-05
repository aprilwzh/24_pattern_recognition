import os
import numpy as np
from tslearn.metrics import dtw
import csv
from tqdm import tqdm
from PIL import Image

def read_keywords():
    keywords_file = "Exercise_4/KWS-test/keywords.tsv"
    keyword_map = {}
    with open(keywords_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            keyword_id = row[0]
            image_filename = f"Exercise_4/images/{row[1]}.jpg"
            svg_filename = f"Exercise_4/KWS-test/locations/{row[1]}.svg"
            keyword_map[keyword_id] = (image_filename, svg_filename)
    return keyword_map

def read_svg(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        words = [line.split("\t")[0] for line in lines]
    return words

def read_image(image_path):
    image = Image.open(image_path)
    return np.array(image)

def calculate_dtw_distance(keyword_image, word_images):
    distances = []
    for word_image in word_images:
        distance = dtw(keyword_image, word_image, global_constraint="sakoe_chiba")
        distances.append(distance)
    return distances

def main():
    keyword_map = read_keywords()
    test_set_file = "Exercise_4/KWS-test/test.tsv"
    test_set = []
    with open(test_set_file, "r") as f:
        for line in f:
            test_set.append(line.strip())

    results = {}

    for keyword_id, filenames in keyword_map.items():
        keyword_image_path, keyword_svg_path = filenames
        keyword_image = read_image(keyword_image_path)

        word_distances = {}

        for test_image_name in test_set:
            word_svg_path = f"Exercise_4/KWS-test/locations/{test_image_name}.svg"
            word_images = read_svg(word_svg_path)
            word_images = [read_image(f"Exercise_4/KWS-test/images/{word_image}.jpg") for word_image in word_images]
            distances = calculate_dtw_distance(keyword_image, word_images)
            mean_distance = np.mean(distances)
            word_distances[test_image_name] = mean_distance

        sorted_distances = sorted(word_distances.items(), key=lambda x: x[1])
        results[keyword_id] = sorted_distances

    with open("output.tsv", "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for keyword_id, distances in results.items():
            row = [keyword_id]
            for pair in distances:
                row.extend(pair)
            writer.writerow(row)

if __name__ == "__main__":
    main()
