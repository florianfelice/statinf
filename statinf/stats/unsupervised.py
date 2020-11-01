import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from ..nonparametrics.kernels import gaussian


class KMeans():
    def __init__(self, k=1, max_iter=100, init='random', random_state=0):
        """
            K-means clustering implementation.

            .. warning:: This function is still under development.
                This is a beta version, please be aware that some functionalitie might not be available.
                The full stable version soon will be released.

            :param k: number of clusters, default is 1.
            :type k: :obj:`int`
            :param max_iter: number of iterations for convergence.
            :type max_iter: :obj:`int`
            :param init: initialization option, options are random or kmeans++ .
            :type init: :obj:`String`
            :param random_state: seed of the random state, default is 0.
            :type random_state: :obj:`int`
            :param labels: labels for each datapoint.
            :type labels: :obj:`numpy.array`
            :param centroids: coordinates of the centroids.
            :type centroids: :obj:`numpy.array`

            :references:
                * Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements of statistical learning (Vol. 1, No. 10).
                  New York: Springer series in statistics.
        """

        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.k = k
        self.labels_ = 0
        self.centroids = 0

    def fit(self, X):
        """
            Fit the model to the data using different initializations (random init or kmeans++)

            :param X: Input data.
            :type X: :obj:`numpy.array`
        """
        # Initialize centroids randomly or using k-means ++
        if type(X) == pd.core.frame.DataFrame:
            data = X.values
        elif type(X) == np.ndarray:
            data = X
        else:
            raise TypeError('X should be either a numpy array or a pandas DataFrame')

        if self.init == 'random':
            centroids = data.copy()
            r = np.random.RandomState(seed=self.random_state)
            r.shuffle(centroids)
            centroids = centroids[:self.k, :]
        elif self.init == 'kmeans++':
            candidates = data.copy()[:, :2]
            # Pick a first random centroid
            random_candidate_index = np.random.randint(0, candidates.shape[0] - 1)
            # Add it to the list of centroids
            centroids = [candidates[random_candidate_index, :]]
            # Remove the picked centroid from the list of potential centroids
            candidates = np.delete(candidates, random_candidate_index, axis=0)
            for k in range(self.k - 1):
                distances = self.get_distance(candidates, np.array(centroids).mean(axis=0).reshape(1, -1))[0]
                # Normalize distances to get the probabilities
                probabilities = distances / distances.sum()
                # Pick new centroid with a probability proportional to the distance
                random_candidate_index = np.random.choice(np.arange(0, candidates.shape[0]), p=probabilities)
                centroids.append(candidates[random_candidate_index, :])
                # Remove the picked centroid from the list of potential centroids
                candidates = np.delete(candidates, random_candidate_index, axis=0)
            centroids = np.array(centroids)

        for _l in range(0, self.max_iter):
            labels_ = self.closest_centroid(data, centroids)  # Find index of closest centroid to each point
            centroids = self.move_centroids(data, labels_, centroids)  # Update the values of the new centroid

        self.labels_ = self.closest_centroid(data, centroids)
        self.centroids = centroids

    def closest_centroid(self, points, centroids):
        """
            Returns an array containing the index to the nearest centroid for each point

            :param points: features of each point.
            :type points: :obj:`numpy.array`
            :param centroids: list of the centroids coordinates.
            :type centroids: :obj:`list`
        """
        distances = self.get_distance(points, centroids)
        return np.argmin(distances, axis=0)

    def get_distance(self, points, centroids):
        """
            Returns the euclidian distance between each point and the centroids.

            :param points: features of each point.
            :type points: :obj:`numpy.array`
            :param centroids: list of the centroids coordinates.
            :type centroids: :obj:`list`
        """
        return np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))

    def move_centroids(self, points, closest, centroids):
        """
            Returns the new centroids assigned from the points closest to them.

            :param points: features of each point.
            :type points: :obj:`numpy.array`
            :param closest: array with the index of closest centroid for each point.
            :type closest: :obj:`numpy.array`
            :param centroids: list of the centroids coordinates.
            :type centroids: :obj:`list`
        """
        return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])

    def silhouette_score(self, X, labels):
        """
            To be added soon
        """
        pass


class GaussianMixture:

    def __init__(self):
        """
            Class for a gaussian mixture model, uses the EM algorithm to fit the model to the data.

            .. warning:: This function is still under development.
                This is a beta version, please be aware that some functionalitie might not be available.
                The full stable version soon will be released.

            :references: * Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.

            :source: * Inspired by: https://github.com/ocontreras309/ML_Notebooks/blob/master/GMM_Implementation.ipynb
        """
        self.clusters = []
        self.likelihoods = None
        self.likelihood = np.Inf
        self.history = None
        self.scores = None

    def _initialize(self, X, k):
        """
            Initialize the clusters using Kmeans

            :param X: Input data.
            :type X: :obj:`numpy.array`

            :param k: Number of (gaussian) clusters.
            :type k: :obj:`int`
        """
        kmeans = KMeans(k=k)
        kmeans.fit(X)
        means = kmeans.centroids
        for _k in range(k):
            self.clusters.append({
                'pi_k': 1 / k,  # Assign equal probabilities
                'mu_k': means[_k],
                'cov_k': np.identity(X.shape[1], dtype=np.float64)
            })

    def _expect(self, X):
        """
            Expectation step, computes the responsibilities using the means and covariances
            computed at the previous maximization step

            :param X: data
            :type X: :obj:`numpy.array`
        """
        totals = np.zeros((X.shape[0], 1), dtype=np.float64)
        for cluster in self.clusters:
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']

            gamma_nk = (pi_k * gaussian(X, mu_k, cov_k)).astype(np.float64)
            cluster['gamma_nk'] = gamma_nk
            totals += gamma_nk

        for cluster in self.clusters:
            cluster['totals'] = totals
            cluster['gamma_nk'] /= totals

    def _maximize(self, X):
        """
            Maximization step, computes the new values of the means and the covariances

            :param X: data
            :type X: :obj:`numpy.array`
        """
        N = X.shape[0]
        for cluster in self.clusters:
            gamma_nk = cluster['gamma_nk']
            Nk = gamma_nk.sum(axis=0)[0]

            cluster['pi_k'] = Nk / N

            mu_k = (1 / Nk) * (gamma_nk * X).sum(axis=0)
            diff = (X - mu_k).reshape(-1, 1)

            cov_k = np.zeros((X.shape[1], X.shape[1]))
            for j in range(X.shape[0]):
                diff = (X[j] - mu_k).reshape(-1, 1)
                cov_k += gamma_nk[j] * np.dot(diff, diff.T)
            cluster['cov_k'] = cov_k / Nk
            cluster['mu_k'] = mu_k

    def _get_likelihood(self, X):
        """
            Returns the total likelihoods over the dataset

            :param X: data
            :type X: :obj:`numpy.array`
        """
        likelihood = []
        sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in self.clusters]))
        self.likelihood = np.sum(sample_likelihoods)
        return self.likelihood

    def fit(self, X, k, n_epochs=100, improvement_threshold=0.0005):
        """
            Fitting function initialized by K-means algorithm.

            :param X: data.
            :type X: :obj:`numpy.ndarray`
            :param K: number of clusters (gaussians).
            :type K: :obj:`int`
            :param n_epochs: number of epochs, default is 100.
            :type n_epochs: :obj:`numpy.ndarray`
            :param improvement_threshold: Threshold from which we consider the likelihood improved, defaults to 0.0005.
            :type improvement_threshold: :obj:`float`, optional
        """
        self._initialize(X, k)
        likelihoods = np.zeros(n_epochs)
        scores = np.zeros((X.shape[0], k))
        history = []

        diff = 1.
        i = 0

        while (i < n_epochs) and (np.abs(diff) >= improvement_threshold):
            i += 1
            clusters_snapshot = []
            for cluster in self.clusters:
                clusters_snapshot.append({
                    'mu_k': cluster['mu_k'].copy(),
                    'cov_k': cluster['cov_k'].copy()
                })
            history.append(clusters_snapshot)
            self._expect(X)
            self._maximize(X)

            _likelihood_old = self.likelihood
            self.likelihood = self._get_likelihood(X)
            diff            = _likelihood_old - self.likelihood
            likelihoods[i]  = self.likelihood
            print(f'Epoch : {i}, Likelihood : {self.likelihood}')

        for i, cluster in enumerate(self.clusters):
            scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)

        self.likelihoods = likelihoods
        self.scores = scores
        self.history = history
