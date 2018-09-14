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


    

    
