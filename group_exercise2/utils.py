# # Analyze the data # #
# Preprocessing:
# # Binarisation? Otsu, Sauvola, Niblack, etc.
# Extract word images from full page (cropping)
# # Bounding box - Easy, but can contain parts of other words
# # Polygon as clipping mask is more precise
import numpy as np
import svg.path


def get_bounding_box(svgpath):
    x_min = np.infty
    y_min = np.infty
    x_max = -np.infty
    y_max = -np.infty
    for section in svgpath:
        if isinstance(section, svg.path.Line):
            xs, ys = section.start.real, section.start.imag
            xe, ye = section.end.real, section.end.imag
            x_min = min(x_min, xs, xe)
            y_min = min(y_min, ys, ye)
            x_max = min(x_max, xs, xe)
            y_max = min(y_max, ys, ye)
        elif isinstance(section, svg.path.QuadraticBezier):
            xs, ys = section.start.real, section.start.imag
            xc, yc = section.end.real, section.end.imag
            xe, ye = section.end.real, section.end.imag
            x_min = min(x_min, xs, xe, xc)
            y_min = min(y_min, ys, ye, yc)
            x_max = min(x_max, xs, xe, xc)
            y_max = min(y_max, ys, ye, yc)
        elif isinstance(section, svg.path.CubicBezier):
            xs, ys = section.start.real, section.start.imag
            xc1, yc1 = section.control1.real, section.control1.imag
            xc2, yc2 = section.control2.real, section.control2.imag
            xe, ye = section.end.real, section.end.imag
            x_min = min(x_min, xs, xe, xc1, xc2)
            y_min = min(y_min, ys, ye, yc1, yc2)
            x_max = min(x_max, xs, xe, xc1, xc2)
            y_max = min(y_max, ys, ye, yc1, yc2)
    return x_min, y_min, x_max, y_max


