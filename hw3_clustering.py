import numpy as np
import pandas as pd
import scipy as sp
import sys
import math
from random import randint

X = np.genfromtxt(sys.argv[1], delimiter = ",") 
centers = [randint(0,len(X)) for i in range(5)]
centerslist = np.array([X[i] for i in centers])