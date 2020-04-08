from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import math, random
import sys
sys.path.append(srcDir)
# import metropolis sampling
from BernMetropolis import *


#-------------------------------------------------------------------------
# pdf construction using kernelmethods
#-------------------------------------------------------------------------
def pn(kernel, x_axis, samples, h1):
    "kernel smoothing"
    
    N = len(samples)
    hn = lambda x: h1 / math.sqrt(x)
    z = np.zeros(len(x_axis))
    #num_samples = len(samples)
    for n in range(N):
        scale = hn(n+1)
        y = kernel((x_axis - samples[n]) / scale) / scale
        z +=  y
    
    return (z / N)
    

#-------------------------------------------------------------------------
# grid subplots
#-------------------------------------------------------------------------
def plot_kernels(kernel, pdf, h_params, N_params, x_range):
    '''
        :params:
            kernel: kernel function
            pdf   : prob density function
            h_params: list of h1
            N_params: list of N (number of samples)
            x_range : range of x values for plotting
    '''
    
    numH = len(h_params)
    numN = len(N_params)
    xmin = x_range[0]
    xmax = x_range[1]
    
    def sample_from(pdf, xmin, xmax, N):
        "sample distribution from pdf"        
        #x = np.random.uniform(xmin, xmax, N)
        NN = min(55555, 5*N)
        samples = metropolis(pdf, theta0=0.0, trajLength=NN)
        # sample N values only
        samples = [x for x in samples if x >= xmin and x <= xmax]
        random.shuffle(samples)
        N = max(len(samples), N)
        return samples[:N]
        
                
    count = 0
    for h1 in h_params:
        for N in N_params:
            x_axis = np.linspace(xmin, xmax, N)
            count += 1
            ax = plt.subplot(numH, numN, count)            
            ax.set_xticks([]) 
            ax.set_yticks([])
            
            # set y label for h
            if (count - 1) % numN == 0:
                plt.ylabel("h = {}".format(h1))

            # set title for N
            if count <= numN:
                plt.title("N = {}".format(N))
                
            # generate samples
            samples = sample_from(pdf, xmin, xmax, N)
            # construct from kernel 
            z = pn(kernel, x_axis, samples, h1)
            plt.plot(x_axis, z, 'k-')
            # plot true distribution
            y_axis = pdf(x_axis)
            plt.plot(x_axis, y_axis, 'r-')
            # plot samples
            plt.plot(samples, np.zeros(len(samples)), 'g.')
            
#-------------------------------------------------------------------------
# testing
#-------------------------------------------------------------------------
mu, sigma = 0, 1
dist = lambda N: np.random.normal(mu, sigma, N)
pdf = norm.pdf

# define kernel function
# kernel = norm.pdf
kernel = lambda x: exp(-x**2/2)/sqrt(2*pi)

h_params = [0.1, 1.0, 10]
N_params = [100, 5000]
x_range = [-4, 3]
plot_kernels(kernel, pdf, h_params, N_params, x_range)


def mixed_pdf(x):
    "mixture of pdf"
    
    def value(x): 
        if x > -3.0 and x < -2.0:
            return x + 3.0
        elif x >= -2.0  and x < -1.0:
            return -2.0 - x + 1
        elif x >= -1.0 and x < 2:
            return 1
        else:
            return 0
        
    if isinstance(x, list) or isinstance(x, np.ndarray):
        return (np.array([value(xx) for xx in x]))
    else:
        return (value(x))
    
h_params = [0.1, 0.3, 1.0]    
N_params = [100, 5000]
x_range = [-4, 4]
plot_kernels(kernel, mixed_pdf, h_params, N_params, x_range)    

# metropolis problem??
samples = metropolis(mixed_pdf, theta0=0.0)
plt.hist(samples)
min(samples), max(samples)

#-------------------------------------------------------------------------  
# histogram
plt.hist(samples, bins=20)

#-------------------------------------------------------------------------
# debug
# np.zeros(len(samples)) + samples
# kernel(0)
# kernel(4)
# hn = lambda x: h1 / math.sqrt(x)
# h1 = 0.2
# hn(10)
# 
# np.random.uniform(-3, 3, 100)

#-------------------------------------------------------------------------
# srcDir = '/home/fra/FraDir/learn/Learnpy/Mypy/bayesian'
# import sys
# from imp import reload
# sys.path.append(srcDir)
# 
# from BernMetropolis import *
# traj = metropolis(pdf, theta0=0.5, trajLength=1000)
# type(traj)
# import random
# random.shuffle(traj)
# traj[-10:]
# plt.hist(traj)


