# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import utils
import numpy as np
import matplotlib.pyplot as plt
import time
import OLD_SVM
from sklearn.preprocessing import StandardScaler


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    folder_name = 'MNIST-full/'
    file_name = 'gt-test.tsv'

    start = time.time()
    test_file_names, test_labels = utils.read_file(folder_name, file_name, train=False)
    end = time.time()

    print(f'time elapsed: {round(end - start, 2)}')
    # print(test_file_names[:10])
    # print(test_labels[:10])

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
    # print(train_file_names[:10])
    # print(train_labels[:10])
    # print(train_file_names.shape)

    start = time.time()
    train_samples = utils.load_files(folder_name, train_file_names)
    # print(train_samples.shape)
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

    linear_svc = OLD_SVM.SVC(kernel="linear")
    rbf_svc = OLD_SVM.SVC(kernel="rbf")
    #
    classifiers = [linear_svc, rbf_svc]

    # print()

    flat_test_samples = np.reshape(test_samples, (len(test_samples), 784), order='C')
    flat_train_samples = np.reshape(train_samples, (len(train_samples), 784), order='C')
    sc = StandardScaler()
    X_test = sc.fit_transform(flat_test_samples)
    X_train = sc.fit_transform(flat_train_samples)
    # print(X_test.shape)
    # print(test_labels.shape)

    start = time.time()
    # X_blobs, y_blobs = SVM.make_blobs(n_samples=300, centers=[[-1.5, -1.5], [1.5, 1.5]])
    # SVM.plot_classifiers_predictions(X_test, test_labels, classifiers)
    linear_svc.fit(X_train, train_labels)
    print('score:', linear_svc.score(X_test, test_labels))
    end = time.time()
    # linear_svc.fit()

    print(f'time elapsed: {round(end - start, 2)}s')
