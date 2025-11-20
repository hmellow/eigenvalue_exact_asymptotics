# -*- coding: utf-8 -*-
"""
Created on 11/18/25 16:41:53

@author: hmellow
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_complex_esd(A):
    eigenvalues = np.linalg.eigvals(A)
    p = A.shape[0]

    plt.figure()
    # plt.title("ESD of non-symmetric random matrix - $\\mathcal{N}(0, 1/p)$ case")
    # plt.title(
    #     "ESD of non-symmetric random matrix - $\\mathbb{P}(A_{ij}=\\frac{1}{\\sqrt{p}})=\\mathbb{P}(A_{ij}=\\frac{-1}{\\sqrt{p}})=\\frac{1}{2}$ case"
    # )
    plt.title(
        "ESD of non-symmetric deterministic matrix - entries are first $p^2$ digits of $\\pi$"
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
