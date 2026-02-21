import numpy as np
import random
import matplotlib.pyplot as plt
import os

def l2Norm(x):
    x = np.asarray(x).ravel()
    output = np.sqrt(np.sum(x**2))
    return output

def l1Norm(x):
    output = (np.sum(np.abs(x)))
    return output

def normalError(x, x_k):
    x = np.asarray(x).ravel()
    x_k = np.asarray(x_k).ravel()
    output = l2Norm(x-x_k)/l2Norm(x)
    return output

def leastSquare(A, B):
    # using normal equations (A.T*A)^-1*A.T*B
    # ATA = A.T @ A
    # ATB = A.T @ B
    # output = np.linalg.inv(ATA) @ ATB
    B = np.asarray(B).ravel()
    output = np.linalg.lstsq(A, B, rcond=None)[0]
    return output
