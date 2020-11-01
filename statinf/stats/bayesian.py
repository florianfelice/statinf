import numpy as np
import matplotlib.pyplot as plt

# TODO: Add priors for the classes for GGM @ybendou

class GGM:

    def __init__(self):
        """
            Gaussian Generative model.
            This class implements a Linear and Quadratic classifiers obtained by assuming Gaussian distributed data and by using Bayes' Theorem.

            .. warning:: This function is still under development.
                This is a beta version, please be aware that some functionalitie might not be available.
                The full stable version soon will be released.
                To be added: priors of the classes

            :formula: .. math:: \\mathbb{P}(A \\mid B) = \\dfrac{\\mathbb{P}(B \\mid A) \\cdot \\mathbb{P}(A)}{\\mathbb{P}(B)}

            :references: * Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.
        """
        self.class_means = None
        self.cov = None
        self.isotropic = True

    def _weighted_norm(self, a, cov=None, norm="euclidian"):
        """
            Returns the norm of a vector using the euclidian or the mahalonobis norm.

            :param cov: Covariance matrix.
            :type  cov: :obj:`numpy.ndarray`
            :param norm: Norm to be used, options are euclidian or mahalonobis, if mahalonobis is specified a proper covariance matrix
                has to be specified too, default is euclidian.
            :type  norm: :obj:`str`
        """
        if norm == "euclidian":
            return np.sum(a**2)
        elif norm == "mahalanobis":
            # assert cov != None, "Specify a covariance matrix"
            return np.dot(np.dot(a, np.linalg.inv(cov)), a)

    def _compute_distance(self, new_data, norm="euclidian"):
        """
            Returns distance between each point and the mean of each class, picks the one that maximizes the likelihood.

            :param new_data: New data features.
            :type  new_data: :obj:`numpy.ndarray`
            :param norm: Norm to be used, options are 'euclidian' or 'mahalonobis', default is 'euclidian'.
            :type  norm: :obj:`str`
        """
        nb_classes = self.class_means.shape[0]
        distances = np.zeros((new_data.shape[0], nb_classes))

        for c in range(nb_classes):  # per class
            if self.isotropic:
                # Add the prior later, ln(p(Ck))
                distances[:, c] = np.apply_along_axis(self._weighted_norm, 1, new_data - self.class_means[c, :], cov=self.cov, norm=norm)
            else:
                distances[:, c] = -np.apply_along_axis(self._weighted_norm, 1, new_data - self.class_means[c, :], cov=self.cov[c, :, :], norm=norm)
                distances[:, c] -= (1 / 2) * np.log(np.linalg.det(self.cov[c, :, :]))
        return distances

    def fit(self, data, labels, nb_classes, isotropic=True):
        """
            Extracts the mean and the covariance of each class, in the isotropic case we assume that all the classes have the same covariance.
            Returns the mean vector for each class, and the estimated covariance.

            :param data: Data features.
            :type  data: :obj:`numpy.ndarray`
            :param labels: Data labels.
            :type  labels: :obj:`numpy.ndarray`
            :param nb_classes: Number of classes in the data.
            :type  nb_classes: :obj:`int`
            :param isotropic: Is an isotropic case or not, meaning the covariance matrix for each class
                is a :math:`(\\sigma^{2}) \\times \\mathbb{I}_{n}` or different (LDA vs QDA).
            :type isotropic: :obj:`bool`
        """
        self.isotropic = isotropic
        class_means = np.zeros((nb_classes, data.shape[1]))

        if not self.isotropic:
            cov = np.zeros((nb_classes, data.shape[1], data.shape[1]))
        else:
            cov = np.cov(data.T)

        for c in range(nb_classes):
            class_means[c, :] = data[labels == c].mean(axis=0)
            if not self.isotropic:
                cov[c, :, :] = np.cov(data[labels == c].T)

        self.class_means = class_means
        self.cov = cov
        return class_means, cov

    def predict(self, new_data, norm="euclidian"):
        """
            Returns predictions for each sample, it affects the labels by finding the closest mean of each class
            using different distances (euclidian, mahalanobis with isotropic or non isotropic covariance)
            For the isotropic case we use a Linear Discriminant classifier, for the non isotropic case we use a
            Quadratic Discriminant classifier.

            :param new_data: New data to evaluate.
            :type  new_data: :obj:`numpy.ndarray`
            :param norm: Norm to be used, options are euclidian or mahalonobis, default is euclidian.
            :type  norm: :obj:`str`
        """
        distances = self._compute_distance(new_data, norm=norm)

        objective = {True: np.argmin, False: np.argmax}  # Maximize or minimize based on the case of isotropic or non isotropic
        predicted_labels = objective[self.isotropic](distances, axis=1)

        return predicted_labels  # this function classifies each point in the test set by affecting it to the

    def predict_proba(self, X, norm="euclidian"):
        """
            Returns the likelihood probability for each class.

            :param X: Data features.
            :type  X: :obj:`numpy.ndarray`
            :param norm: Norm to be used, options are euclidian or mahalonobis, default is euclidian.
            :type  norm: :obj:`str`
        """
        distances = self._compute_distance(X, norm=norm)
        likelihood = np.exp(distances - distances.max(axis=1)[:, np.newaxis])

        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def plot_decision_boundary(self, X, labels, norm="euclidian", grid_size=100, *args):
        """
            Plots the predictions on the entire dataset as well as the decision boudaries for each class

            :param X: Data features.
            :type  X: :obj:`numpy.ndarray`
            :param labels: Labels of each point in the training set.
            :type  labels: :obj:`numpy.ndarray`
            :param norm: Norm to be used, options are euclidian or mahalonobis, default is euclidian.
            :type  norm: :obj:`str`
            :param grid_size: Size of the square grid to be plotted.
            :type  grid_size: :obj:`int`
        """
        x = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), grid_size)  # extent of the grid on the x axis
        y = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), grid_size)  # extent of the grid on the y axis
        [xx, yy] = np.meshgrid(x, y)
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()], norm=norm)
        Z = Z.reshape(grid_size, grid_size)
        plt.figure(*args)
        plt.scatter(X[:, 0], X[:, 1], c=labels)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.6)
        plt.show()
