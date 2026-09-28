#coding:utf8

import numpy as np
import pandas as pd
import scipy
import scipy.stats

#https://docs.scipy.org/doc/scipy/reference/stats.html

dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)

# Question 1
print("Question 1")
import matplotlib.pyplot as plt
x = np.arange(-5, 6)
y = np.where(x == 0, 1, 0)

plt.stem(x, y)
plt.title("Loi de Dirac")
plt.savefig("img/dirac.png")
plt.close()
x = np.arange(1, 7)
y = scipy.stats.randint.pmf(x, 1, 7)

plt.stem(x, y)
plt.title("Loi uniforme discrète")
plt.savefig("img/uniforme_discrete.png")
plt.close()
x = np.arange(0, 11)
y = scipy.stats.binom.pmf(x, 10, 0.5)

plt.stem(x, y)
plt.title("Loi binomiale")
plt.savefig("img/binomiale.png")
plt.close()
x = np.arange(0, 20)
y = scipy.stats.poisson.pmf(x, 5)

plt.stem(x, y)
plt.title("Loi de Poisson")
plt.savefig("img/poisson.png")
plt.close()
x = np.arange(1, 21)
y = 1 / ((x + 1) ** 2)
y = y / y.sum()

plt.stem(x, y)
plt.title("Loi de Zipf-Mandelbrot")
plt.savefig("img/zipf_mandelbrot.png")
plt.close()
x = np.linspace(-5, 5, 1000)
y = scipy.stats.norm.pdf(x, 0, 1)

plt.plot(x, y)
plt.title("Loi normale")
plt.savefig("img/normale.png")
plt.close()
x = np.linspace(0.01, 5, 1000)
y = scipy.stats.lognorm.pdf(x, 1)

plt.plot(x, y)
plt.title("Loi log-normale")
plt.savefig("img/lognormale.png")
plt.close()
x = np.linspace(-1, 2, 1000)
y = scipy.stats.uniform.pdf(x, 0, 1)

plt.plot(x, y)
plt.title("Loi uniforme")
plt.savefig("img/uniforme.png")
plt.close()
x = np.linspace(0.01, 20, 1000)
y = scipy.stats.chi2.pdf(x, 5)

plt.plot(x, y)
plt.title("Loi du Chi-deux")
plt.savefig("img/chi2.png")
plt.close()
x = np.linspace(1, 5, 1000)
y = scipy.stats.pareto.pdf(x, 2)

plt.plot(x, y)
plt.title("Loi de Pareto")
plt.savefig("img/pareto.png")
plt.close()

# Question 2
print("Question 2")
def moyenne(distribution):
    return np.mean(distribution)

def ecart_type(distribution):
    return np.std(distribution)
distributions = {
    "Dirac": np.where(np.arange(-5, 6) == 0, 1, 0),
    "Uniforme discrète": scipy.stats.randint.rvs(1, 7, size=1000),
    "Binomiale": scipy.stats.binom.rvs(10, 0.5, size=1000),
    "Poisson": scipy.stats.poisson.rvs(5, size=1000),
    "Normale": scipy.stats.norm.rvs(0, 1, size=1000),
    "Log-normale": scipy.stats.lognorm.rvs(1, size=1000),
    "Uniforme": scipy.stats.uniform.rvs(0, 1, size=1000),
    "Chi-deux": scipy.stats.chi2.rvs(5, size=1000),
    "Pareto": scipy.stats.pareto.rvs(2, size=1000)
}

for nom, distribution in distributions.items():
    print(nom, "Moyenne :", moyenne(distribution), "Ecart-type :", ecart_type(distribution))
