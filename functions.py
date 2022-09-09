import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

def rand_weights(n):
    '''Devuelve las ponderaciones aleatorias en un vector de N dimensiones.'''
    k = np.random.rand(n)
    return k / sum(k)

def random_portfolio(returns):
    '''Devuelve la media, desviación estandar, ponderación de cada activo
    y el ratio de sharpe correspondiente a cada portafolio.'''
    p = np.asmatrix(np.mean(returns, axis=0))
    w = np.asmatrix(rand_weights(returns.shape[1]))
    C = np.asmatrix(returns.cov())
    weights = w.tolist()
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    sharper = (mu - 0.0429)/sigma
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma, weights, sharper

def summary_portfolios(n, returns):
    '''Devuelve un dataframe con N portafolios ordenado por:
    INDEX | MEAN | STANDDEV |SHARPER | W1 | W2 | W3...'''
    means, stds, weights, sharper = np.column_stack([
        random_portfolio(returns) 
        for _ in range(n)
    ])
    summary = pd.DataFrame(weights.tolist(), columns = '%'+returns.columns)
    summary.insert(loc = 0, column='Means', value = means)
    summary.insert(loc = 1, column='StanDev', value = stds)
    summary.insert(loc = 2, column='SharpeR', value = sharper)
    return summary

def best_portfolio(summary):
    '''Devuelve un DataFrame con el portafolio más optimo
    según el ratio de sharpe.'''
    best = summary.iloc[[summary['SharpeR'].astype('float').idxmax()]]
    return best
