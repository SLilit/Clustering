import numpy as np
import pandas as pd
import scipy as sp
import sys
import math
from random import randint

X = np.genfromtxt(sys.argv[1], delimiter = ",") 
def KMeans(data, centers):
    n = [0]*5
    mean = np.zeros(shape = (5,8))
    new_data = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for j in range(len(data)):
        if np.isnan(data[j].any()):
            cluster = randint(0,4)
        else:
            min_c = float('inf')
            for k in range(5):
                var = data[j] - centers[k]
                var_t = var.reshape((-1,1))
                var = var.reshape((1,8))
                norm = np.dot(var,var_t)
                if norm < min_c:
            #norm = np.linalg.norm(data[j] - centers[k])
            #if norm < min_c:
                    cluster = k
                    min_c = norm
        
        n[cluster] += 1
        mean[cluster] += data[j]
        new_data[cluster].append(data[j][:])
    
    means = [m/n for m,n in zip(mean, n)]
    
    
    return means
  

centers = [randint(0,len(X)) for i in range(5)]
centerslist = np.array([X[i] for i in centers])
for i in range(10):
    filename = "centroids-" + str(i+1) + ".csv"       #"i" would be each iteration
    np.savetxt(filename, centerslist, delimiter=",")
    centerslist = KMeans(X, centerslist)


    

    
def EMGMM(data, pi, mu, sigma):
    
    sigma_det = np.linalg.det(sigma)
    new_data = {0: [], 1: [], 2: [], 3: [], 4: []}  
    print ('pis: ', pi)
    count = 0
    for x in data:
        if np.isnan(x.any()):
            count += 1
            pass
        pre = [0.]*5
        mx = float('-inf')
        mn = float('inf')
        #c = 0
        for k in range(5):
            
            var = x - mu[k]
            var_t = var.reshape((-1,1))
            var = var.reshape((1,8))
            inv = np.linalg.inv(sigma[k])
            e = math.exp(-0.5*np.dot(np.dot(var,inv),var_t))
            f = abs(sigma_det[k])
            f = math.pow(f,-0.5)
            p = pi[k]
            f =math.pow(p,0.5)*f*e
            pre[k] = f
            if f > mx:
                mx = f
                c = k
            elif f < mn:
                mn = f
        
        phi = pre[c]/sum(pre[:])                  
        new_data[c].append([x, phi])
                           
    pis = np.array([float(len(new_data[k]))/len(data) for k in range(5)])
    means = np.zeros(shape = (5,8))
    sigmas = np.array([[[0.]*8]*8]*5)
                           
    for k in range(5):
                           
        mean = np.zeros(shape = (1,8))
        for x in new_data[k]:
            mean += x[0]*x[1]
        means[k] = mean[:]/len(new_data[k])
        
        sigma = np.zeros(shape = (8,8))                   
        for x in new_data[k]:
            var = x[0] - means[k]
            var_t = var.reshape((-1,1))
            var = var.reshape((1,8))
            sigma += x[1]*np.dot(var_t,var)
        sigmas[k] = sigma[:]/len(new_data[k]) 
                           
    return pis, means, sigmas    
    
    
