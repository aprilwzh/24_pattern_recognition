import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.svm import SVC


def plot_classifiers_predictions(x, y, classifiers):
    """
    Plots the decision regions and support vectors of the classifiers
    fit on x and y.

    Args:
        x: Data matrix of shape [num_train, num_features]
        y: Labels of shape [num_train]
        classifiers: A list of classifier objects

    """
    fig, axes = plt.subplots(ncols=len(classifiers), nrows=1, figsize=(16, 8))

    for classifier, axis in zip(classifiers, axes.flat):
        # Fit the classifier to the data
        classifier.fit(x, y)
        print(classifier.n_features_in_)
        # Plot the decision regions
        plot_decision_regions(x, y, clf=classifier, legend=2, ax=axis,
                              filler_feature_values=range(classifier.n_features_in_))

        # Plot the support vectors
        x_support = x[classifier.support_]
        axis.scatter(x_support[:, 0], x_support[:, 1], c='red', marker='x', s=100, label='Support Vectors')

        axis.set_title('{}, accuracy = {:.2f}'.format(
            classifier.__class__.__name__, classifier.score(x, y)))

    plt.show()


# linear_svc = SVC(kernel="linear")
# rbf_svc = SVC(kernel="rbf")
#
# classifiers = [linear_svc, rbf_svc]
#
# X_blobs, y_blobs = make_blobs(n_samples=300, centers=[[-1.5, -1.5], [1.5, 1.5]])
# plot_classifiers_predictions(X_blobs, y_blobs, classifiers)
