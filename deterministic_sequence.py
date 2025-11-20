# -*- coding: utf-8 -*-
"""
Created on 11/18/25 15:27:37

@author: hmellow
"""
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp, pi
from utils import *


def generate_deterministic_matrix(p):
    """
    Create a deterministic matrix based on the first p^2 digits of pi

    :param p:
    :return:
    """
    mp.dps = p**2
    pi_digits = str(pi).replace(".", "")
    pi_digits = [int(d) for d in pi_digits]
    A = np.array(pi_digits).reshape((p, p))
    return A


p = 1000


def main():
    B = generate_deterministic_matrix(p)
    empirical_mean = np.mean(B)
    empirical_variance = np.var(B)

    A = (B - empirical_mean) / (np.sqrt(p * empirical_variance))

    plot_complex_esd(A)


if __name__ == "__main__":
    main()
