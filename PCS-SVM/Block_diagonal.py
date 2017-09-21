# -*- coding: utf-8 -*-
"""
Block_diagonal.py implements the generative block-diagonal clustering model 
as described in [1]. 

[1] Junxiang Chen and Jennifer Dy, "A Generative Block-Diagonal Model for 
Clustering", Proc. of the Conference on Uncertainty in Artificial Intelligence
2016 (UAI2016), New York City, NY, USA, June 2016.

The implementation allows the data to be given in the form of either feature 
matrix or similaliry matrix. We provide examples for both cases.
See test_synthetic() and iris() for detail.
"""
__author__              =           "Junxiang Chen"
__email__               =           "jchen@ece.neu.edu"


import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.special import digamma
from scipy.special import betaincinv

from scipy.stats import beta

from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.cluster import SpectralClustering
from sklearn import datasets

def test_synthetic(seed = 0, alpha = .2):
    """
    Train the model with synthetic similarity matrix.
    """
    np.random.seed(seed)
    
    # Generating synthetic similarity matrix.
    n_data = 100
    K = 3
    X = np.random.dirichlet([alpha] * K, n_data)
    W = np.dot(X, X.T)
    label = np.argmax(X,1)
    
    # Fit the model
    M = block_diagonal(W, K = K, similarity_matrix_given = True, label = label)
    M.fit()    

    # Print the NMI.                
    print(M.nmi())


    # Show the resulted similarity matrix.
    M.show_matrix()
    plt.show()



def iris():
    """
    Train the model with the iris data.
    """
    # load the dataset.
    iris = datasets.load_iris()
    
    # Normalize the data such that it has zero-mean and unit-variance.
    data = iris.data
    print (data)
    return
    data = (data - data.mean(0)) / data.std(0)

    
    M = block_diagonal(data, label = iris.target, K = 3)
    M.fit()    

    # Print the NMI.                
    #print(M.nmi())
    
    # Show the resulted similarity matrix.
    M.show_matrix()
    plt.show()

                                                                                                                                                                                                                                                                                                
class block_diagonal:
    def __init__(self, X, K = 2, similarity_matrix_given = None, 
        degree_normalize = True, 
        mu_zeta = 15., sigma2_zeta = 1., alpha_zeta = 8e4 , beta_zeta =  2e4, 
        mu_eta = 0., sigma2_eta = 1., alpha_eta = 1e3, beta_eta = 9e3, 
        label = None):

        """
        X represents the data. Data can be provided as either a feature matrix
        or the similarity matrix.
        
        K represents the number of clusters.
        
        If similarity_matrix_given is True, it indicates that X is a similarity 
        matrix. If similarity_matrix_given is False, it indicates that X is a 
        feature matrix and the similarity matrix will be derived from the 
        feature matrix. If similarity_matrix_given is None, then we check 
        whether the matrix is squared and determine whether X is a feature 
        matrix or a similarity matrix.
        
        If degree_normalize is True, we normalize the similarity matrix with 
        respect to the degrees.
        
        mu_zeta, sigma2_zeta, alpha_zeta, beta_zeta, mu_eta, sigma2_eta,
        alpha2_eta, alpha_eta, beta_eta are the parameters of the model. See the
        paper for details.
        
        label represents the ground-truth label of the dataset.
        :rtype: object
        """          
    
        [self.N, self.D] = X.shape
        
        if (similarity_matrix_given is None and self.N != self.D) \
                or similarity_matrix_given == False:

            self.X = X        
            
            # We compute the normalized euclidean distance and then derive the 
            # similarity matrix from the feature matrix.
            dist = pdist(X, 'euclidean')
            dist_median = np.median(dist)
            eucl_dist= squareform(dist)
            dist_normal = - eucl_dist[:,:] ** 2 / ( 2 * dist_median ** 2)
            self.W = np.exp(dist_normal)
            #print(self.W)

            if degree_normalize:
                # We normalize the similarity matrix using its degree matrix
                np.fill_diagonal(self.W, 0)
                inv_sqrt_D = np.diag( 1. / np.sqrt(self.W.sum(0)))
                self.W = np.dot(np.dot(
                            inv_sqrt_D, self.W),
                        inv_sqrt_D)
        else:
            # If the similarity matrix is already given, we do not need to 
            # compute it.
            self.X = None
            self.D = None
            self.W = X
        
        # We make sure the similarity matrix has values between 0 and 1.
        eye = np.eye(self.N, dtype = bool)
        self.W = (self.W - self.W[~eye].min()) / \
                    (self.W[~eye].max() - self.W[~eye].min())
        np.fill_diagonal(self.W, 1)
                    
        
        # Note that the update equations are functions of log(W_ij) and 
        # log(1 - W_ij). We precompute these values.
        self.log_W = np.log(self.W + 1e-300) # log(W_ij)
        self.log_WI = np.log(1 - self.W[:,:] + 1e-300) # log(1 - W_ij)

                
        
        self.K = K
        
        self.mu_zeta = mu_zeta
        self.sigma2_zeta = sigma2_zeta
        
        self.alpha_zeta = alpha_zeta
        self.beta_zeta = beta_zeta
            

        self.mu_eta = mu_eta
        self.sigma2_eta = sigma2_eta
        self.alpha_eta = alpha_eta
        self.beta_eta = beta_eta
            
        self.label = label
        self.initialize()

    def initialize(self, spectral_initialization = False):
        """
        If spectral_initialization is True, we initialize the model using 
        spectral clustering. Otherwise, we randomly initialize the model.
        """
        self.alpha_k = np.random.gamma(1,1,self.K)
        self.beta_k = np.random.gamma(1,1,self.K)
        
        
        self.alpha_0 = np.random.gamma(1,1)
        self.beta_0 = np.random.gamma(1,1)

        tmp = [1.] * self.K

        self.pi_nk = np.random.dirichlet ( tmp , self.N)

        '''
        if spectral_initialization:        
            M = SpectralClustering(n_clusters = self.K, 
                affinity = 'precomputed')
            z = M.fit_predict(self.W)            
                    
            self.pi_nk = np.zeros([self.N, self.K])
            
            for k in range(self.K):
                self.pi_nk[z == k,k ] = 1
        '''

    def log_L_Theta_k(self, theta_k, k):
        """
        Returns the negative lowerbound as a function of Theta_k. 
        """
        Ez_nk = np.atleast_2d(self.pi_nk[:,k])
        
        tmp_prod = Ez_nk.T * Ez_nk
            
        tmp_prod -= np.diag(np.diag(tmp_prod))
        
        alpha_k = theta_k[0] ** 2
        beta_k = theta_k[1] ** 2
        res = 0
        res += -np.log(alpha_k + beta_k) - (np.log(alpha_k + beta_k) \
                    - self.mu_zeta)**2 / (2 * self.sigma2_zeta)
        res += (self.alpha_zeta - 1) * np.log(alpha_k) \
            + (self.beta_zeta - 1) * np.log(beta_k) \
            - (self.alpha_zeta + self.beta_zeta - 2) * np.log(alpha_k + beta_k)
        res += (tmp_prod * (
                    (alpha_k - 1) * self.log_W[:,:] 
                   + (beta_k - 1) * self.log_WI[:,:] )
                ). sum() / 2. 
                
        res += - tmp_prod.sum() / 2. * betaln(alpha_k, beta_k)

        res += ( betaln(alpha_k, beta_k)
                    - (alpha_k - 1) * digamma(alpha_k) 
                    - (beta_k - 1) * digamma(beta_k)
                    + (alpha_k + beta_k - 2) * digamma(alpha_k + beta_k)
                )
                                                
        return -res
        
    def log_L_Theta_0(self, Theta_0):
        """
        Returns the negative lowerbound as a function of Theta_0. 
        """
        tmp_prod = np.zeros([self.N, self.N])
        
        for k in range(self.K):
            z_nk = np.atleast_2d(self.pi_nk[:,k])
            tmp_prod += z_nk.T * z_nk
            
        tmp_prod = 1 - tmp_prod

        tmp_prod -= np.diag(np.diag(tmp_prod))
        
        alpha_0 = Theta_0[0] ** 2
        beta_0 = Theta_0[1] ** 2

        res = 0
        res += - np.log(alpha_0 + beta_0) - \
            (np.log(alpha_0 + beta_0) - self.mu_eta)**2 / (2 * self.sigma2_eta)
        res += (self.alpha_eta - 1) * np.log(alpha_0) + \
            (self.beta_eta - 1) * np.log(beta_0) \
            - (self.alpha_eta + self.beta_eta - 2) * np.log(alpha_0 + beta_0)
       
        res += (tmp_prod * ( 
                    (alpha_0 - 1)* self.log_W + (beta_0 - 1) * self.log_WI )
                ).sum()/2.
        
        res += - tmp_prod.sum() / 2. * betaln(alpha_0, beta_0)
                        
        res += ( betaln(alpha_0, beta_0)
                    - (alpha_0 - 1) * digamma(alpha_0) 
                    - (beta_0 - 1) * digamma(beta_0)
                    + (alpha_0 + beta_0 - 2) * digamma(alpha_0 + beta_0)
                )
        return -res
        
  

    def lowerbound(self):
        """
        Returns the lowerbound.
        """
        tau_k = self.alpha_k + self.beta_k
        mu_k = self.alpha_k / tau_k

        tau_0 = self.alpha_0 + self.beta_0
        mu_0 = self.alpha_0 / tau_0
        
        res = 0
        
        res += ( - np.log(tau_k) - (np.log(tau_k) - self.mu_zeta) ** 2\
                        / (2. * self.sigma2_zeta) ).sum()
        res += ((self.alpha_zeta - 1) * np.log(mu_k) 
                    + (self.beta_zeta - 1) * np.log(1 - mu_k)).sum()

        res +=- np.log(tau_0) - (np.log(tau_0) - self.mu_eta) ** 2 \
                    / (2. * self.sigma2_eta)
        res += (self.alpha_eta - 1) * np.log(mu_0) \
                + (self.beta_eta - 1) * np.log(1 - mu_0)
        

        tmp_sum = np.zeros([self.N, self.N])                        
        
        for k in range(self.K):
            z_ik = np.atleast_2d(self.pi_nk[:,k])
            
            tmp_prod = z_ik.T * z_ik
            tmp_sum += tmp_prod
            
            tmp_prod -= np.diag(np.diag(tmp_prod))
            
            res += (tmp_prod * (
                        (self.alpha_k[k] - 1) * self.log_W 
                        + (self.beta_k[k] - 1) * self.log_WI)
                    ).sum() / 2.
            res += - tmp_prod.sum() / 2. \
                    * betaln(self.alpha_k[k], self.beta_k[k])
        
        tmp = 1 - tmp_sum
        tmp -= np.diag(np.diag(tmp))
        
        res += (tmp * (
                    (self.alpha_0 - 1) * self.log_W 
                    + (self.beta_0 - 1) * self.log_WI)
                ).sum() / 2.
        res += - tmp.sum() / 2. * betaln(self.alpha_0, self.beta_0)
        
        
                        
        res += (self.pi_nk * np.log(1. / (self.K + 1))).sum()
        
        
       
        res += ( betaln(self.alpha_k, self.beta_k)
                    - (self.alpha_k - 1) * digamma(self.alpha_k) 
                    - (self.beta_k - 1) * digamma(self.beta_k)
                    + (self.alpha_k + self.beta_k - 2) 
                        * digamma(self.alpha_k + self.beta_k)
                ).sum()
        
        res += ( betaln(self.alpha_0, self.beta_0)
                    - (self.alpha_0 - 1) * digamma(self.alpha_0) 
                    - (self.beta_0 - 1) * digamma(self.beta_0)
                    + (self.alpha_0 + self.beta_0 - 2) 
                        * digamma(self.alpha_0 + self.beta_0)
                )

        #print (self.pi_nk)
        tmp = - self.pi_nk * np.log(self.pi_nk)
        res += tmp[ np.logical_not( np.isnan(tmp))].sum()

        return res
    
                                                                           
    def variational_inference(self, n_iter = 20, threshold = 1e-5):
        """
        The variational inference for the model.
        
        n_iter is the number of iteration.
        threshold determines the convergence of the process.
        """
        for i_iter in range(n_iter):
            #print ( i_iter, '/', n_iter)
            current_lowerbound = self.lowerbound()
            
            # Equation (12)
            for k in range(self.K):
                res = minimize(self.log_L_Theta_k, 
                    [np.sqrt(self.alpha_k[k]), np.sqrt(self.beta_k[k])],
                    (k,) , method = "CG").x
                self.alpha_k[k] = res[0] ** 2
                self.beta_k[k] = res[1] ** 2
                
            # Equation (13)
            res = minimize(self.log_L_Theta_0, 
                [np.sqrt(self.alpha_0), np.sqrt(self.beta_0)], method = "CG").x


            self.alpha_0 = res[0] ** 2
            self.beta_0 = res[1] ** 2
                                
            # Equations (16) and (17)
            shuffled_ind = list(range(self.N))
            np.random.shuffle(shuffled_ind)
            for i in shuffled_ind:
                
                log_pi_nk = np.zeros(self.K)
                
                for k in range(self.K):
                    Ez_jk = self.pi_nk[:,k].copy()
                    Ez_jk[i] = 0
                    
                    log_pi_nk[k] += (Ez_jk * (
                        (self.alpha_k[k] - self.alpha_0) * self.log_W[i,:] 
                         + (self.beta_k[k] - self.beta_0) * self.log_WI[i,:])
                        ).sum()
                    log_pi_nk[k] += Ez_jk.sum() \
                            * (betaln(self.alpha_0, self.beta_0) \
                        - betaln(self.alpha_k[k], self.beta_k[k]))
                        
                log_pi_nk -= log_pi_nk.max()
                self.pi_nk[i,:] = np.exp(log_pi_nk)
                self.pi_nk[i,:] /= self.pi_nk[i,:].sum()
            

            # Check convergence.            
            new_lowerbound = self.lowerbound()
            #print ("Lowerbound:", new_lowerbound)
            
            if (new_lowerbound - current_lowerbound)\
                    / np.abs(current_lowerbound) < threshold:
                break
            else:
                current_lowerbound = new_lowerbound

                                                        
    def fit(self,n_init = 10):
        """
        Repeat random initialization and variational inference several times.
        
        n_init is the number of random initialization.
        """
        max_lowerbound = -1e300
        for i in range(n_init):
            if i > n_init / 2:
            # Half of the initializations are given by spectral clustering
                self.initialize(spectral_initialization = True)
            else:
            # The remainings are random initializations
                self.initialize(spectral_initialization = False)
            self.variational_inference()
            
            #Pick the solution with the largest lowerbound.
            lowerbound = self.lowerbound()
            '''
            Error occured here - change here:
            add the defination of Theta and Pi
            time: 2017/9/17
            '''
            #-------------------------------------------------
            Theta = [self.alpha_k.copy(), 
                    self.beta_k.copy(), self.alpha_0, self.beta_0]
            pi = self.pi_nk.copy()
            #-------------------------------------------------
            if lowerbound>max_lowerbound:
                Theta = [self.alpha_k.copy(), 
                    self.beta_k.copy(), self.alpha_0, self.beta_0]
                pi = self.pi_nk.copy()
                max_lowerbound = lowerbound
            #print (i, "Lowerbound:", lowerbound)
        self.alpha_k, self.beta_k, self.alpha_0, self.beta_0 = Theta
        self.pi_nk = pi
                    

    def show_matrix(self):
        """
        Show the similarity matrix whose indices are sorted according to the 
        clustering results.
        """
        z = np.argmax(self.pi_nk,1)
        #print(z)

        ind = z.argsort()
        plt.figure()
        plt.imshow(self.W[ind,:][:,ind],
            interpolation = "none", vmin = 0, vmax = 1)
        plt.colorbar()

        font = {'weight' : 'light',
        'size'   : 15}

        plt.rc('font', **font)




    def nmi(self):
        """
        Return the Normalized Mutual Information (NMI) of the clustering 
        solution with respect to the ground-truth labels given.
        """
        z = np.argmax(self.pi_nk,1)
        return nmi(z, self.label)
        
    def returnsW(self):
        """
        Return the similarity Matrix.

        """
        return self.W


    def s_new(self,sim):
        #pp = sim**(self.alpha_k-1)*(1-sim)**(self.beta_k-1)/(beta(self.alpha_k,self.beta_k))
        p=beta.cdf(sim,self.alpha_k, self.beta_k)
        #print(p)
        p_old=max(p)
        #print(p_old)
        return p_old
        #pin=betaincinv(self.alpha_0,self.beta_0,pp)
        #print(pin)

    def s_old(self,sim):
        #pp = sim**(self.alpha_k-1)*(1-sim)**(self.beta_k-1)/(beta(self.alpha_k,self.beta_k))
        p_new=beta.cdf(sim,self.alpha_0, self.beta_0)
        #print(p_new)
        #pin=betaincinv(self.alpha_0,self.beta_0,pp)
        #print(pin)
        return p_new
