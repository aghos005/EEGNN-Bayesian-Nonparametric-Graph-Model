import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import fsolve
import scipy.special
from scipy.special import gamma
import tensorflow as tf
from scipy.stats import poisson
tf.config.run_functions_eagerly(True)
#V is number of observed nodes
class CompletelyRandomMeasureWithLevy:
    def __init__(self, kappa, base_measure):
        # Initialize a Completely Random Measure with Levy measure.

        # Parameters:
        # - kappa: Levy measure parameter (a function that generates samples for dθ)
        # - base_measure: Base measure (a function that generates samples for dw)
        
        self.kappa = kappa
        self.base_measure = base_measure

    def draw_measure(self, size):
        # Draw a sample from the CRM with Levy measure.

        # Parameters:
        # - size: Number of samples to draw

        # Returns:
        # - theta_samples: Sampled values of θ
        # - weight_intensity_measure: Weight intensity measure function

        # Sample θ and weights
        theta_samples = self.kappa(size)
        weights = self.base_measure(size)
        weights /= np.sum(weights)

        # Return both the samples and the weight intensity measure function
        return theta_samples, self.weight_intensity_measure_function(theta_samples, weights), weights

    def weight_intensity_measure_function(self, theta_samples, weights):
        # Construct the weight intensity measure function
        
        # Parameters:
        # - theta_samples: Sampled values of θ
        # - weights: Sampled weights corresponding to θ_samples

        # Returns:
        # - intensity_measure: Weight intensity measure function

        # Define the weight intensity measure function using interpolation
        def intensity_measure(theta):
            # Find the weight associated with the given theta using linear interpolation
            weight = np.interp(theta, theta_samples, weights, left=0.0, right=0.0)
            return weight

        return intensity_measure
    def laplace_transform(self, s, theta_values, weights):
        intensity_function = self.weight_intensity_measure_function(theta_values, weights)
        integrand = lambda theta: np.exp(-s * theta) * intensity_function(theta)
        laplace_transform, _ = quad(integrand, 0, np.inf)
        return laplace_transform

    def density_function(self, theta_values, weights):
        # Integrate the Laplace transform over the entire sample space
        laplace_integral, _ = quad(lambda s: self.laplace_transform(s, theta_values, weights), 0, np.inf)

        # Define the density function
        def density_function(theta):
            return np.exp(-laplace_integral) * self.weight_intensity_measure_function(theta_values, weights)(theta)

        return density_function

def GEM(alpha, num_clusters):
    
    #Generate a sample from the GEM distribution.

   # Parameters:
    #- alpha: Concentration parameter
    #- num_clusters: Number of clusters to generate

    #Returns:
    #- A sample from the GEM distribution
    beta_params = np.ones(num_clusters) * alpha
    beta_samples = np.random.beta(1, beta_params)
    pi = beta_samples * np.concatenate(([1], np.cumprod(1 - beta_samples[:-1])))

    return pi


#alpha=? #Concentration parameter
#num_clusters = 10

 #Draw a sample from GEM distribution
#K is number of clusters
#PI is distribution for clusters
class GammaProcess:
    def __init__(self, base_measure, alpha):
        # Initialize a Gamma Process with a given base measure and shape parameter alpha.
        # Parameters:
        # - base_measure: Array representing the base measure
        # - alpha: Shape parameter of the Gamma distribution
        self.base_measure = base_measure
        self.alpha = alpha

    def draw_process(self, num_points):
        # Draw a sample from the Gamma process.
        # Parameters:
        # - num_points: Number of points to draw
        # Returns:
        # - Points sampled from the Gamma process

        # Draw points from a uniform distribution
        points = np.sort(np.random.uniform(0, 1, num_points))

        # Sample from the discrete base measure
        base_measure_values = np.interp(points, np.linspace(0, 1, len(self.base_measure)), self.base_measure)

        # Compute increments based on the Poisson distribution
        increments = np.random.poisson(base_measure_values * self.alpha)

        # Compute the process values as the cumulative sum of increments
        process_values = np.cumsum(increments)

        return process_values
class CategoricalDistribution:
    def __init__(self, probabilities):
        
        #Initialize the categorical distribution with given probabilities.

        #param probabilities: A list or array of probabilities for each category.
        
        self.probabilities = np.array(probabilities)
        self.categories = np.arange(len(probabilities))

        # Check if probabilities sum to 1
        if not np.isclose(np.sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.")

    def sample(self, size=1):
        #Generate random samples from the categorical distribution.

        #:param size: Number of samples to generate.
        #:return: Array of samples.
        
        return np.random.choice(self.categories, size=size, p=self.probabilities)

def ind_tf(condition):
    return tf.where(condition, 1, 0)

kappa = lambda size: tf.random.uniform(shape=(size,), minval=0, maxval=1)
gamma_scale = 1.0

def gamma_base_measure(size):
    return tf.random.gamma(shape=(size,), alpha=2.0, beta=gamma_scale)
def clustweight(V,alpha): 
    return GEM(alpha,V+1)
def weight(K,V,w_0):
    W=[]
    W.append(w_0)
    for i in range(K): 
        #Sociability Parameters is sampled from a gamma process with base measure W_0
        
        g= GammaProcess(w_0,2)
        w=g.draw_process(V+1)
        W.append(w)
    matrix_W= np.array(W)
    return matrix_W
def sample(K,V,alpha): 
    kappa = lambda size: np.random.uniform(low=0, high=1, size=size)
    gamma_scale = 1.0
    
    # Create an instance of CompletelyRandomMeasureWithLevy
    
    crm_with_levy = CompletelyRandomMeasureWithLevy(kappa, gamma_base_measure)
    
    # Draw a sample from the CRM and obtain the weight intensity measure function
    W_0, weight_intensity_measure, weights= crm_with_levy.draw_measure(V+1)
    
    # Calculate the density function
    density_function = crm_with_levy.density_function(W_0, weights)
    W=weight(K,V,W_0)
    PI= clustweight(V,alpha)
    Z = np.ndarray((V,V)) #i,j<---i+1,j+1
    for i in range(V):
        for j in range(V):
        
            #Edge multiplicity is sampled from a Poisson Distribution
                
            pois_param=0
                
            for k in range(1,K+1):
                pois_param = pois_param +  PI[k]*W[k,i+1]*W[k,j+1]
                
            Z[i,j]= np.random.poisson(pois_param)
    #Cluster membership array
    Clust= np.zeros((V,V,int(np.sum(Z))))    
    lambda_clust= 0 
        #probailities for the Categorical Distribution
    probab=[]
        
    for i in range(1,K+1): 
        lambda_clust= lambda_clust + PI[i]*sum(W[i])*sum(W[i])
        
    for i in range(1,K+1): 
        probab.append(PI[i]*sum(W[i])*sum(W[i])/lambda_clust)
    prob = np.array(probab)
    cat_dist= CategoricalDistribution(probab)
    for i in range(V):
        for j in range(V):
            for k in range(int(Z[i,j])):
                if((Z[i,j]+Z[j,i])!=0):
                    #cluster membership for edge i,j
                    Clust[i,j,k]= cat_dist.sample()[0]+1
    row_sums = np.sum(W, axis=1)
    return Clust,PI,Z,W,W_0,weight_intensity_measure,  density_function,row_sums
def MCMC(K,V,Clust,PI,Z,W,W_0,weight_intensity_measure,  density_function,row_sums,alpha):
  #Monte Carlo Markov Chain 
    N = np.zeros((K, V))
    for k in range(K):
        for i in range(V): 
            for j in range(V):
                l=0
                for _ in Clust[i,j]: 
                    l+=1
                for m in range(l):
                    N[k,i]= N[k,i]+ ind_tf(Clust[i,j,m]==k+1) + ind_tf(Clust[j,i,m]==k+1)

#Step1: Hmailtonian Monte Carlo
    def log_posterior(w,w_0,z,c):
        log_posterior=0 
        for k in range(K): 
            for i in range(V): 
                for j in range(V): 
                    log_posterior += np.log((gamma(w_0)+1e-8)/(gamma(w_0+ np.sum(N[k]))+1e-8)) + np.log((gamma(w[i]+ N[k,i])+1e-8)/(gamma(w[i])+1e-8)) + np.log(weight_intensity_measure(w[j])+1e-8)
        log_posterior += np.log( density_function(w_0 - np.sum(w))+1e-8)
        return log_posterior

# Function to compute gradients of log-posterior
    def compute_gradients(w, w_0, z, c):
        epsilon=1e-8
        gradients = []
        for i in range(V):
            gradient_i = 0 
            for k in range(K): 
                for j in range(V): 
                    gradient_i+= scipy.special.digamma(N[k,j]+ w[j]) - scipy.special.digamma(w[j])
            for m in range(V): 
                gradient_i += (np.log(w[m]+epsilon + 1e-5)- np.log(w[m]+ 1e-5))/epsilon
            gradient_i+= np.log(w_0 - np.sum(w) + 1e-5)
            gradients.append(gradient_i)
        return np.array(gradients)

# Leapfrog integration
    def leapfrog_integration(w, p, step_size, num_steps, w_0, z, c):
        for _ in range(num_steps):
            p += 0.5 * step_size * compute_gradients(w, w_0, z, c)
            w += step_size * p
            p += 0.5 * step_size * compute_gradients(w, w_0, z, c)
        return w, p

# Metropolis acceptance step
    def metropolis_acceptance(log_posterior_current, log_posterior_proposed):
        ratio = np.exp(log_posterior_proposed - log_posterior_current)
        return tf.random.uniform(()) < ratio

    # HMC sampling
    def hmc_sampling(w_0, initial_w, z, c, step_size, num_steps):
        w = initial_w
        p = np.random.normal(size=np.shape(w))

    
        grad_log_posterior_current = compute_gradients(w, w_0, z, c)
    
        # Leapfrog integration
        proposed_w, proposed_p = leapfrog_integration(w, p, step_size, num_steps, w_0, z, c)
    
        grad_log_posterior_proposed = compute_gradients(proposed_w, w_0, z, c)
    
        log_posterior_current = -log_posterior(w, w_0, z, c)
        log_posterior_proposed = -log_posterior(proposed_w, w_0, z, c)
    
        if metropolis_acceptance(log_posterior_current, log_posterior_proposed):
            proposed_w = w
            return np.array(w)
        else: 
            return w
    
    num_samples = 1000
    step_size = 0.0001
    num_steps = 1000
    arr= W_0[1:]
    arr = np.exp(hmc_sampling(row_sums[0],arr,Z,Clust,step_size, num_steps))
    W_0[1:]=arr
    W[0]= W_0
    #Step2: Update W_k for k>0
    conc_param1=[]  
    for i in range(0,V):   
            conc_param1.append(W_0[1:][i]+ np.sum(N[:,(i-1)])+ 1e-10)
    
    conc_param1 = np.array(conc_param1)
  
    for i in range(1,K+1):
        x= np.random.dirichlet(conc_param1)
        a= x*row_sums[i]
        
        W[i][1:]=a
        

    #Step3: Update the cluster distribution 
    conc_param2= [] 
    conc_param2.append(alpha)
    for i in range(K): 
        conc_param2.append(np.sum(N[k])+1)
    PI = np.random.dirichlet(conc_param2)
    
    
    #Step4: Update the cluster membership array 
    count=0
    C= np.zeros((V,V,int(np.sum(Z))))
    for i in range(V):
        for j in range(V): 
            prob= []
            for k in range(K+1): 
                print(PI[k]*W[k,i]*W[k,j])
                prob.append(PI[k]*W[k,i]*W[k,j])
            prob=np.array(prob)
            prob= prob/np.sum(prob)
            cls= np.arange(0,K+1)
        for m in range(int(Z[i,j])): 
            C[i,j,m]= np.random.choice(cls,p=prob)
            if(C[i,j,m]==0): 
                C[i,j,m]=K+1
                count+=1 
    if(count>0): 
        K+=1
    
    #Step5: update the unobserved z_ij
    for i in range(V): 
        for j in range(V):
            sum=0
            for k in range(1,K+1):
              sum+=PI[k]*(W[k,i])*(W[k,j])
            print(sum)
            Z[i,j]= poisson.rvs(mu=lambda_param) + 1
    #Step6:update w_0 and w_k using Metroplis Hastings
    
    
    # Function to compute the log-posterior with respect to w_k (you need to define this based on your model)
    def log_posterior_w_k(w_k,w_0,pi,z,c):
        return (2*np.sum(N[k]) + w_0)*np.log(w_k+ 1e-5) -w_k - pi[k]*w_k*w_k
    
    # Function to update w_k using Metropolis-Hastings
    def metropolis_hastings_update_w_k(w_k_current, w_0,pi,z,c , proposal_std):
    # Propose a new value for w_k by adding random noise
        w_k_proposed = w_k_current + np.random.normal(0, proposal_std)
    
    # Calculate acceptance ratio based on the log-posterior
        log_posterior_current = log_posterior_w_k(w_k_current, w_0, pi , z, c)
        log_posterior_proposed = log_posterior_w_k(w_k_proposed,w_0,pi,z,c )
        acceptance_ratio = np.exp(log_posterior_proposed - log_posterior_current)
    
    # Accept or reject the proposed value of w_k
        if np.random.uniform(0, 1) < acceptance_ratio:
            return w_k_proposed  # Accepted
        else:
            return w_k_current  # Rejected
        
        # Set the proposal standard deviation
        proposal_std = 0.1
        for i in range(1,K+1): 
        # Perform Metropolis-Hastings updates for w_k
            num_iterations = 10
            for _ in range(num_iterations):
                row_sums[i] = metropolis_hastings_update_w_k(row_sums[i],row_sums[0],PI,Z,Clust , proposal_std)
    
    def log_posterior_w_0(w_0, w, pi,c,z):
        log_rs= np.log(w[1:]) 
        return np.log(density_function(w_0)+ 1e-5) + w_0*sum(log_rs) -K*np.log(gamma(w_0)+1e-5)
        
        # Function to update w_k using Metropolis-Hastings
        def metropolis_hastings_update_w_0(w_0, w, pi,c,z, proposal_std):
        # Propose a new value for w_k by adding random noise
            w_0_proposed = w_0 + np.random.normal(0, proposal_std)
            
            # Calculate acceptance ratio based on the log-posterior
            log_posterior_current = log_posterior_w_0(w_0_current, w, pi,c,z)
            log_posterior_proposed = log_posterior_w_k(w_0_proposed, w, pi,c,z)
            acceptance_ratio = np.exp(log_posterior_proposed - log_posterior_current)
            
            # Accept or reject the proposed value of w_k
            if np.random.uniform(0, 1) < acceptance_ratio:
                return w_0_proposed  # Accepted
            else:
                return w_0_current  # Rejected
            
            # Example usage:
            # Initialize parameter and data
            
            # Set the proposal standard deviation
            proposal_std = 0.1
            
            # Perform Metropolis-Hastings updates for w_
            num_iterations = 1000
            for _ in range(num_iterations):
                row_sums[0] = metropolis_hastings_update_w_0(row_sums[0], row_sums[1:], PI,Clust,Z, proposal_std)
    return Z,C