# -*- coding: utf-8 -*-
import numpy as np
import math

""" 
This file contains function (algorithm) for dissimilarity measure
"""
Euclidean_Distance = lambda x, y: math.sqrt(Squared_Euclidean_Distance(x, y))
Manhattan_Distance = lambda x, y: sum(np.abs(np.subtract(x, y)))
Squared_Euclidean_Distance = lambda x, y : sum(np.power(np.subtract(x, y), 2))
Cosine_dis = lambda x,y : 1- sum(np.multiply(x,y))
Cosine = lambda x,y : sum(np.multiply(x,y))
def Max(x, axis=1):
    return x / x.sum(axis=axis, keepdims=True)



