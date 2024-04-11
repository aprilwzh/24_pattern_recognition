# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import utils
import numpy as np
import matplotlib.pyplot as plt
import time

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folder_name = 'MNIST-full/'
    file_name = 'gt-test.tsv'

    start = time.time()
    test_file_names, test_labels = utils.read_file(folder_name, file_name, train=False)
    end = time.time()

    print(f'time elapsed: {round(end - start, 2)}')
    print(test_file_names[:10])
    print(test_labels[:10, 0])

    start = time.time()
    test_samples = utils.load_files(folder_name, test_file_names)
    end = time.time()

    print(f'time elapsed: {round(end - start, 2)}')

    # idxs = np.flatnonzero(np.equal(labels, cls))
    idxs = np.random.choice(range(0, len(test_file_names)), 10, replace=False)
    for i, idx in enumerate(idxs):
        # plt_idx = y * samples_per_class + i + 1
        plt.subplot(1, 10, i+1)
        plt.imshow(test_samples[idx])
        plt.axis('off')
        # plt.title(cls)
    plt.show()

    file_name = 'gt-train.tsv'

    start = time.time()
    train_file_names, train_labels = utils.read_file(folder_name, file_name, train=True)
    end = time.time()

    print(f'time elapsed: {round(end - start, 2)}')
    print(train_file_names[:10])
    print(train_labels[:10, 0])

    start = time.time()
    train_samples = utils.load_files(folder_name, train_file_names)
    end = time.time()

    print(f'time elapsed: {round(end - start, 2)}')

    # idxs = np.flatnonzero(np.equal(labels, cls))
    idxs = np.random.choice(range(0, len(train_file_names)), 10, replace=False)
    for i, idx in enumerate(idxs):
        # plt_idx = y * samples_per_class + i + 1
        plt.subplot(1, 10, i + 1)
        plt.imshow(train_samples[idx])
        plt.axis('off')
        # plt.title(cls)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
