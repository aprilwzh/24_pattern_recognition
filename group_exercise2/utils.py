# # Analyze the data # #
# Preprocessing:
# # Binarisation? Otsu, Sauvola, Niblack, etc.
# Extract word images from full page (cropping)
# # Bounding box - Easy, but can contain parts of other words
# # Polygon as clipping mask is more precise
import cv2
import matplotlib.image
import numpy as np
import svg.path
import xml.etree.ElementTree as ET


def get_bounding_box(svgpath):
    x_min = np.infty
    y_min = np.infty
    x_max = -np.infty
    y_max = -np.infty
    for section in svgpath:
        xs = [x_min, x_max]
        ys = [y_min, y_max]
        if isinstance(section, svg.path.Line):
            xs.extend((section.start.real, section.end.real))
            ys.extend((section.start.imag, section.end.imag))
        elif isinstance(section, svg.path.QuadraticBezier):
            xs.extend((section.start.real, section.control.real, section.end.real))
            ys.extend((section.start.imag, section.control.imag, section.end.imag))
        elif isinstance(section, svg.path.CubicBezier):
            xs.extend((section.start.real, section.control1.real, section.control2.real, section.end.real))
            ys.extend((section.start.imag, section.control1.imag, section.control2.imag, section.end.imag))

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
    return x_min, y_min, x_max, y_max


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
    for i, polygon in polygons:
        path = svg.path.parse_path(polygon)

        x_min, y_min, x_max, y_max = get_bounding_box(svgpath=path)

        word_image = np.array(img[x_min:x_max, y_min:y_max])
        word_boxes.append(path)
        word_images.append(word_image)
    return word_boxes, word_images


def binarize_images(images):
    binary_images = []
    for image in images:
        image = cv2.resize(image, (100, 100))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        binary_images.append(np.where(gray_image >= 128, 1, 0)[0])

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
    left =  0
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



