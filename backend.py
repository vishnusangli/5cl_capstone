# %%
import numpy as np
import scipy.optimize as sc
import pandas as pd
import matplotlib.pyplot as plt

# %%
def rms(y, yfit):
    return np.sqrt(np.sum((y-yfit)**2))
def chisqr(y, yfit, y_unc):
    """
    Returns chi squared \\
    y: theoretical, original values \\
    yfit: Observed/fit values \\
    """
    return sum(np.divide((y-yfit)^2, y_unc))

def red_chisqr(y, yfit):
    return np.divide(chisqr(y, yfit), len(y))

def equivalent_unc(x, y, x_unc, y_unc, func, p0):
    """
    Linear equivalent unc generation
    Equation 5.3.9 in Statistics review
    """
    yfit, yfunc = weighted_fit(x, y, func, p0)
    total_unc = np.power(np.power(y_unc, 2) + np.power(yfit,2) + np.power(x_unc, 2), 0.5)
    return total_unc

class Functions:
    def give_func(func, *args):
        """
        Returns the desired function with parameters
        """
        return lambda x: func(x, *args)
    
    def linear1(x, a):
        """
        linear without intercept
        """
        return (a*x)
    def linear2(x, a, b):
        """
        linear with intercept
        """
        return (a*x) + b

    def cos_sqr(x, a, b, c):
        val = (a * np.cos(x + b)) + c
        return val


def weighted_fit(x, y, func, p0, sigma = [], abs_sigma = False, maxfev = 1000):
    """
    Gives least-squares fit \\
    x: input x-array \\
    y: input y-array \\
    func: Desired fit function \\
    p0: initial values for fucntion coeffs 
    """
    if len(sigma) == 0:
        popt, pcov = sc.curve_fit(func, x, y, p0, maxfev = maxfev)
    else:
        popt, pcov = sc.curve_fit(func, x, y, p0, sigma = sigma, absolute_sigma = True, maxfev = maxfev)

    print('Unweighted fit parameters:', popt)
    print('Covariance matrix:'); print(pcov)

    yfit = func(x, *popt)
    print('rms error in fit:', rms(y, yfit))
    print("Fit Params and their uncertainties:")
    temp = [(popt[i], np.sqrt(pcov[i, i])) for i in range(len(popt))]
    print(temp)
    return yfit, Functions.give_func(func, *popt), temp

def linear_weight_lstsqr(x, y, y_unc):
    w = np.divide(1, y_unc^2)
    delta = sum(w) * sum(w * (x^2) + sum(w * x)^2)

    m = np.divide(sum(w)*sum(w * x * y) - np.multiply(sum(w * x), sum(w, y)), delta)
    c = sum(w * y) + np.multiply(m , sum(w * x))/sum(w)
# %%
