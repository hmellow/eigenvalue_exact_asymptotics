# -*- coding: utf-8 -*-
"""
Created on 11/16/25 16:14:36

@author: hmellow

Analyzing the empirical spectral distribution of a non-symmetric random matrix
"""


import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_random_matrix(p, mean, variance):
    A = np.random.normal(mean, variance, size=(p, p))
    return A


def generate_scaled_binary_random_matrix(p):
    A = np.random.binomial(n=1, p=0.5, size=(p, p))  # Bernoulli trials with p=0.5
    A = (A * 2) - 1  # {0, 1} -> {-1, 1}
    A = A / np.sqrt(p)  # {-1, 1} -> {1/-sqrt(p), 1/sqrt(p)}
    return A


def plot_eigenvalue_histogram(A, gamma, variance):
    eigenvalues = np.linalg.eigvals(A)
    p = A.shape[0]

    plt.figure()
    # plt.title("ESD of non-symmetric random matrix - $\\mathcal{N}(0, 1/p)$ case")
    plt.title(
        "ESD of non-symmetric random matrix - $\\mathbb{P}(A_{ij}=\\frac{1}{\\sqrt{p}})=\\mathbb{P}(A_{ij}=\\frac{-1}{\\sqrt{p}})=\\frac{1}{2}$ case"
    )
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.plot(
        eigenvalues.real,
        eigenvalues.imag,
        ".",
        color="blue",
        label="Eigenvalues",
    )

    # Plot the theoretical semicircle
    r = 1
    theta = np.linspace(0, 2 * np.pi, 100)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.plot(
        x,
        y,
        color="red",
        linewidth=2,
        label="Theoretical distribution from circular law",
    )

    plt.legend(loc="best")
    plt.show()


p = 2000
gamma = 1  # Scaling


def main():
    # A = generate_gaussian_random_matrix(p, mean=0, variance=np.sqrt(1 / p**gamma))
    A = generate_scaled_binary_random_matrix(p)
    plot_eigenvalue_histogram(A, gamma, np.sqrt(1 / p**gamma))


if __name__ == "__main__":
    main()
