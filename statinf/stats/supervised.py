import numpy as np
import matplotlib.pyplot as plt

# TODO: Add priors for the classes for GGM #ybendou

class GGM(object):
    """
    Gaussian Generative model
    This class implements a Linear and Quadratic classifiers obtained by assuming Gaussian distributed data and by using Bayes' Theorem.
    To be added : priors of the classes
    """
    def __init__(self):
        """
            :param class_means: means of each class.
            :type  class_means: :obj:`numpy.array` 

            :param cov: covariance matrix of each class.
            :type  cov: :obj:`numpy.array` 
        """
        self.class_means = None
        self.cov = None
        self.isotropic = True

    def weighted_norm(self,a,cov = None, norm = "euclidian"):
        """
            Returns the norm of a vector using the euclidian or the mahalonobis norm.
            
            :param cov: covariance matrix.
            :type  cov: :obj:`numpy.array` 

            :param norm: norm to be used, options are euclidian or mahalonobis, if mahalonobis is specified a proper covariance matrix has to be specified too, default is euclidian.
            :type  norm: :obj:`String` 
        """
        if norm == "euclidian" :
            return np.sum(a**2)
        elif norm == "mahalanobis" : 
            # assert cov != None, "Specify a covariance matrix"
            return np.dot(np.dot(a,np.linalg.inv(cov)),a)

    def fit(self,training_data,training_labels,nb_classes,isotropic = True):
        """
            Extracts the mean and the covariance of each class, in the isotropic case we assume that all the classes have the same covariance.
            Returns the mean vector for each class, and the estimated covariance.

            :param training_data: data features.
            :type  training_data: :obj:`numpy.array` 

            :param training_labels: data labels.
            :type  training_labels: :obj:`numpy.array` 

            :param nb_classes: number of classes in the data.
            :type  nb_classes: :obj:`int`
            
            :param isotropic: boolean value for if it's an isotropic case or not, meaning the covariance matrix for each class is a (sigma**2)*Identity or different (LDA vs QDA).
            :type  training_data: :obj:`numpy.array` 
        """
        self.isotropic = isotropic
        class_means = np.zeros((nb_classes,training_data.shape[1]))

        if not self.isotropic : 
            cov = np.zeros((nb_classes,training_data.shape[1],training_data.shape[1]))
        else : 
            cov = np.cov(training_data.T)

        for c in range(nb_classes):
            class_means[c,:] = training_data[training_labels == c].mean(axis = 0)
            if not self.isotropic :  
                cov[c,:,:] = np.cov(training_data[training_labels == c].T)                
    
        self.class_means = class_means
        self.cov = cov
        return class_means, cov


    def predict(self,test_data,norm = "euclidian"):
        """
            Returns predictions for each sample, it affects the labels by finding the closest mean of each class
            using different distances (euclidian, mahalanobis with isotropic or non isotropic covariance) 
            For the isotropic case we use a Linear Discriminant classifier, for the non isotropic case we use a 
            Quadratic Discriminant classifier.

            :param test_data: test data features.
            :type  test_data: :obj:`numpy.array` 

            :param norm: norm to be used, options are euclidian or mahalonobis, default is euclidian.
            :type  norm: :obj:`String` 
        """
        distances = self.compute_distance(test_data,norm = norm)

        objective = {True : np.argmin,False : np.argmax} #Maximize or minimize based on the case of isotropic or non isotropic
        predicted_labels = objective[self.isotropic](distances,axis = 1)
        
        return predicted_labels # this function classifies each point in the test set by affecting it to the
    
    def predict_proba(self,X,norm = "euclidian"):
        """
            Returns the likelihood probability for each class.
            
            :param X: data features.
            :type  X: :obj:`numpy.array` 

            :param norm: norm to be used, options are euclidian or mahalonobis, default is euclidian.
            :type  norm: :obj:`String`
        """
        distances = self.compute_distance(X,norm = norm)
        likelihood = np.exp(distances - distances.max(axis = 1)[:,np.newaxis])

        return likelihood/likelihood.sum(axis = 1)[:,np.newaxis]

    def compute_distance(self,test_data,norm = "euclidian"):
        """
            Returns distance between each point and the mean of each class, picks the one that maximizes the likelihood.
            
            :param test_data: test data features.
            :type  test_data: :obj:`numpy.array` 

            :param norm: norm to be used, options are euclidian or mahalonobis, default is euclidian.
            :type  norm: :obj:`String`
        """
        nb_classes = self.class_means.shape[0]
        distances = np.zeros((test_data.shape[0],nb_classes))
        
        for c in range(nb_classes): # per class
            if self.isotropic :
                distances[:,c] = np.apply_along_axis(self.weighted_norm,1,test_data-self.class_means[c,:],cov = self.cov,norm = norm) # Add the prior later, ln(p(Ck))
            else:
                distances[:,c] = -np.apply_along_axis(self.weighted_norm,1,test_data-self.class_means[c,:],cov = self.cov[c,:,:],norm = norm) - 1/2*np.log(np.linalg.det(self.cov[c,:,:]))
        return distances
    def plot_decision_boundary(self,X,labels,norm = "euclidian",grid_size = 100):
        """
            Plots the predictions on the entire dataset as well as the decision boudaries for each class
            
            :param X: data features.
            :type  X: :obj:`numpy.array` 

            :param labels: labels of each point in the training set.
            :type  labels: :obj:`numpy.array` 

            :param norm: norm to be used, options are euclidian or mahalonobis, default is euclidian.
            :type  norm: :obj:`String`

            :param grid_size: size of the square grid to be plotted.
            :type  grid_size: :obj:`int`
        """
        x = np.linspace(np.min(X[:,0]),np.max(X[:,0]),grid_size) # extent of the grid on the x axis
        y = np.linspace(np.min(X[:,1]),np.max(X[:,1]),grid_size) # extent of the grid on the y axis
        [xx,yy] = np.meshgrid(x,y)
        Z = self.predict(np.c_[xx.ravel(),yy.ravel()],norm = norm)
        Z = Z.reshape(grid_size,grid_size)
        plt.scatter(X[:,0],X[:,1],c = labels)
        plt.contourf(xx,yy,Z,cmap = plt.cm.RdBu,alpha = 0.6)
