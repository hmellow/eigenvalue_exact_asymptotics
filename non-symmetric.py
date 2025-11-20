# -*- coding: utf-8 -*-
"""
Created on 11/16/25 16:14:36

@author: hmellow

Analyzing the empirical spectral distribution of a non-symmetric random matrix
"""


import numpy as np
import matplotlib.pyplot as plt
from utils import *


def generate_gaussian_random_matrix(p, mean, variance):
    A = np.random.normal(mean, variance, size=(p, p))
    return A


def generate_scaled_binary_random_matrix(p):
    A = np.random.binomial(n=1, p=0.5, size=(p, p))  # Bernoulli trials with p=0.5
    A = (A * 2) - 1  # {0, 1} -> {-1, 1}
    A = A / np.sqrt(p)  # {-1, 1} -> {1/-sqrt(p), 1/sqrt(p)}
    return A


p = 2000
gamma = 1  # Scaling


def main():
    # A = generate_gaussian_random_matrix(p, mean=0, variance=np.sqrt(1 / p**gamma))
    A = generate_scaled_binary_random_matrix(p)
    # plot_complex_esd(A, gamma, np.sqrt(1 / p**gamma))
    plot_copmlex_esd(As)


if __name__ == "__main__":
    main()
