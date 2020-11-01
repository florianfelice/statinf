from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import getpass
import sys

if sys.platform == 'darwin':
    sys.path.append(f"/Users/{getpass.getuser()}/Documents/statinf/")
elif sys.platform in ['linux', 'linux1', 'linux2']:
    sys.path.append(f"/home/{getpass.getuser()}/statinf/")
else:
    sys.path.append(f"C:/Users/{getpass.getuser()}/Documents/statinf/")

from statinf.stats.bayesian import GGM
from statinf.stats.unsupervised import GaussianMixture
from statinf.regressions.LinearModels import LinearBayes


def test_ggm(norm="mahalanobis", isotropic=True):
    if isotropic:
        if norm == "euclidian":
            n_samples = 100
            X, labels = make_blobs(n_samples=[n_samples, n_samples, n_samples], cluster_std=[0.5, 0.5, 0.5],
                                   centers=None, n_features=2, random_state=0)
        elif norm == "mahalanobis":
            covariance_matrix = [[-1, -4], [2, 6]]
            nb_samples = 1000
            X_class_1 = sp.stats.multivariate_normal.rvs(mean=[5, 5], cov=covariance_matrix, size=nb_samples)
            X_class_2 = sp.stats.multivariate_normal.rvs(mean=[-1, 11], cov=covariance_matrix, size=nb_samples)
            X_class_3 = sp.stats.multivariate_normal.rvs(mean=[0, 0], cov=covariance_matrix, size=nb_samples)

            X = np.concatenate((X_class_1, X_class_2, X_class_3), axis=0)
            labels = np.concatenate([np.zeros(nb_samples), np.ones(nb_samples), 2 * np.ones(nb_samples)], 0)
    else:
        nb_samples = 500
        X, labels = make_blobs(n_samples=[nb_samples, nb_samples, nb_samples], cluster_std=[0.5, 1, 2],
                               centers=[[2, 2], [-5, -5], [2, 2]], n_features=3, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.33)
    nb_classes = 3

    classifier = GGM()
    means, covariance = classifier.fit(X_train, y_train, nb_classes, isotropic=isotropic)
    _ = classifier.predict(X_test, norm=norm)
    classifier.plot_decision_boundary(X, labels, norm=norm)


test_ggm(isotropic=True, norm="euclidian")
test_ggm(isotropic=True, norm="mahalanobis")
test_ggm(isotropic=False)


def test_gaussian_mixture():

    iris = datasets.load_iris()
    X = iris.data

    gmm = GaussianMixture()
    K = 3
    gmm.fit(X, K)
    return gmm.clusters

test_gaussian_mixture()

def test_linear_bayesian(true_sigma=10):
    """
        Test function for the Linear_bayesian class
    """

    # Generate linear data and add noise
    def generate_linear_data(n_samples=50, bias=0, weights=5, true_sigma=10):

        true_w = np.array([[bias, weights]]).T
        X = 10 * np.random.rand(n_samples, 1) - 1

        phi = np.hstack((np.ones((X.shape[0], 1)), X))

        noise = true_sigma * (1 - 2 * np.random.rand(n_samples, 1))
        y = phi @ true_w + noise

        return X, y

    X, y = generate_linear_data(true_sigma=true_sigma)
    plt.figure()
    plt.scatter(X, y.reshape(1, -1)[0])

    lb = LinearBayes()
    lb.fit(X, y, true_sigma=true_sigma)
    plt.figure()
    lb.plot_posterior_line(X, y)
    plt.figure()
    lb.plot_weight_distributions()

test_linear_bayesian(true_sigma=10)
