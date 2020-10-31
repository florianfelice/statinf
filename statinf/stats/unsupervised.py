import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import random 


class KMeans():
    def __init__(self,
                K=1,
                max_iter=100,
                init='random',
                random_state=0
                ):
        """
            Kmeans implementation
            :param K: number of clusters, default is 1.
            :type K: :obj:`int`
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
        """

        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.K = K
        self.labels_ = 0
        self.centroids = 0

    def fit(self,X):
        """
            Fit the model to the data using different initializations (random init or kmeans++) 
            :param X: data.
            :type X: :obj:`numpy.array`
        """
        #Initialize centroids randomly or using k-means ++ 
        if type(X) == pd.core.frame.DataFrame:
            data = X.values
        elif type(X) == np.ndarray:
            data = X
        else : 
            raise TypeError('X should be either a numpy array or a pandas DataFrame')
        if self.init == 'random':
            centroids = data.copy()
            r = np.random.RandomState(seed=self.random_state)
            r.shuffle(centroids)
            centroids = centroids[:self.K,:]
   
        if self.init == 'kmeans++':
            candidates = data.copy()[:,:2]
            random_candidate_index = np.random.randint(0,candidates.shape[0]-1) # Pick a first random centroid
            centroids = [candidates[random_candidate_index,:]] # Add it to the list of centroids
            candidates = np.delete(candidates,random_candidate_index,axis=0) # Remove the picked centroid from the list of potential centroids
            for k in range(self.K-1):
                distances = self.get_distance(candidates,np.array(centroids).mean(axis=0).reshape(1,-1))[0]
                probabilities =  distances / distances.sum() #Normalize distances to get the probabilities
                random_candidate_index = np.random.choice(np.arange(0,candidates.shape[0]),p=probabilities) # Pick new centroid with a probability proportional to the distance
                centroids.append(candidates[random_candidate_index,:])
                candidates = np.delete(candidates,random_candidate_index,axis=0) # Remove the picked centroid from the list of potential centroids
            centroids = np.array(centroids)
                        
        for l in range(0,self.max_iter) : 
            labels_ = self.closest_centroid(data, centroids) #Find index of closest centroid to each point
            centroids = self.move_centroids(data, labels_, centroids) #Update the values of the new centroid
        
        self.labels_ = self.closest_centroid(data, centroids)
        self.centroids = centroids

    def closest_centroid(self,points, centroids):
        """
            Returns an array containing the index to the nearest centroid for each point
            :param points: features of each point.
            :type points: :obj:`numpy.array`
            :param centroids: list of the centroids coordinates.
            :type centroids: :obj:`list`
        """
        distances = self.get_distance(points,centroids)
        return np.argmin(distances, axis=0)

    def get_distance(self,points,centroids):
        """
            Returns the euclidian distance between each point and the centroids.
            :param points: features of each point.
            :type points: :obj:`numpy.array`
            :param centroids: list of the centroids coordinates.
            :type centroids: :obj:`list`
        """
        return np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))

    def move_centroids(self,points, closest, centroids):
        """
            Returns the new centroids assigned from the points closest to them.
            :param points: features of each point.
            :type points: :obj:`numpy.array`
            :param closest: array with the index of closest centroid for each point.
            :type closest: :obj:`numpy.array`
            :param centroids: list of the centroids coordinates.
            :type centroids: :obj:`list`
        """
        return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
    
    def silhouette_score(self,X,labels):
        """
            To be added soon
        """
        pass

class GaussianMixture():
    """
        Class for a gaussian mixture model, uses the EM algorithm to fit the model to the data.
        Inspired by https://github.com/ocontreras309/ML_Notebooks/blob/master/GMM_Implementation.ipynb
    """
    def __init__(self):
        """
            :param clusters: list of dictionnaries, each dictionnary represents a cluster with 4 parameters (mean, cov, gamma the responsibility, pi the weight of each gaussian)
            :type clusters: :obj:`list`
            
            :param likelihoods: list of likelihoods per class for each point.
            :type likelihoods: :obj:`numpy.array`
        
            :param history: history snapshots of the clusters for each iteration.
            :type history: :obj:`list`

            :param scores: list of log of responsibilities per class for each point.
            :type scores: :obj:`list`
        """
        self.clusters = []
        self.likelihoods = None
        self.history = None
        self.scores = None

    def _initialize(self,X,K):
        """
            Initialize the clusters using Kmeans
            
            :param X: data
            :type X: :obj:`numpy.array`
            
            :param K: number of clusters (gaussians).
            :type K: :obj:`int`
        """
        kmeans = KMeans(K = K)
        kmeans.fit(X)
        means= kmeans.centroids
        for k in range(K):
            self.clusters.append({
                'pi_k':1/K, # Assign equal probabilities
                'mu_k':means[k],
                'cov_k':np.identity(X.shape[1],dtype = np.float64)
            })
        
    def _expect(self,X):
        """
            Expectation step, computes the responsibilities using the means and covariances 
            computed at the previous maximization step
            :param X: data
            :type X: :obj:`numpy.array`
        """
        totals = np.zeros((X.shape[0], 1), dtype=np.float64)
        for cluster in self.clusters :
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']

            gamma_nk = (pi_k*self.gaussian(X,mu_k,cov_k)).astype(np.float64)
            cluster['gamma_nk'] = gamma_nk
            totals += gamma_nk

        for cluster in self.clusters:
            cluster['totals'] = totals
            cluster['gamma_nk'] /= totals


    def _maximize(self,X):
        """
            Maximization step, computes the new values of the means and the covariances : 
            source is Machine Learning : A Probabilistic Perspective [Murphy 2012-08-24] page 351 
            :param X: data
            :type X: :obj:`numpy.array`
        """
        N = X.shape[0]
        for cluster in self.clusters:
            gamma_nk = cluster['gamma_nk']        
            Nk = gamma_nk.sum(axis = 0)[0]

            cluster['pi_k'] = Nk/N 

            mu_k = (1/Nk)*(gamma_nk*X).sum(axis = 0)
            diff = (X-mu_k).reshape(-1, 1)

            cov_k = np.zeros((X.shape[1],X.shape[1]))
            for j in range(X.shape[0]):
                diff = (X[j] - mu_k).reshape(-1, 1)
                cov_k += gamma_nk[j] * np.dot(diff, diff.T)
            cluster['cov_k'] = cov_k/Nk
            cluster['mu_k'] = mu_k
    
    def gaussian(self,X,mu,cov):
        """
            Returns the pdf of a Gaussian kernel
            :param X: data
            :type X: :obj:`numpy.array`
            :param mu: mean of the gaussian
            :type mu: :obj:`numpy.array`
            :param cov: covariance matrix
            :type cov: :obj:`numpy.array`
        """
        X_centered = X-mu
        n = X.shape[1]
        return np.diagonal((1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5))*np.exp(-0.5*np.dot(np.dot(X_centered, np.linalg.inv(cov)), X_centered.T))).reshape(-1,1)

    def get_likelihood(self,X):
        """
            Returns the total likelihoods over the dataset
            :param X: data
            :type X: :obj:`numpy.array`
        """
        likelihood = []
        sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in self.clusters]))
        return np.sum(sample_likelihoods)

    def fit(self,X,K,n_epochs = 100):
        """
            Training function
            :param X: data.
            :type X: :obj:`numpy.array`
            :param K: number of clusters (gaussians).
            :type K: :obj:`int`
            :param n_epochs: number of epochs, default is 100.
            :type n_epochs: :obj:`numpy.array`
        """
        self._initialize(X,K)
        likelihoods = np.zeros(n_epochs)
        scores = np.zeros((X.shape[0],K))
        history = []

        for i in range(n_epochs):
            clusters_snapshot = []
            for cluster in self.clusters:
                clusters_snapshot.append({
                    'mu_k': cluster['mu_k'].copy(),
                    'cov_k': cluster['cov_k'].copy()
                })
            history.append(clusters_snapshot)
            self._expect(X)
            self._maximize(X)
            
            likelihood = self.get_likelihood(X)
            likelihoods[i] = likelihood
            print(f'Epoch : {i}, Likelihood : {likelihood}')

        for i,cluster in enumerate(self.clusters):
            scores[:,i] = np.log(cluster['gamma_nk']).reshape(-1)
    
        self.likelihoods = likelihoods
        self.scores = scores
        self.history = history
    