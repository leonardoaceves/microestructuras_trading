
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

comision, capital = 0.00125, 1000000
naftrac = pd.read_excel('C:\\naftrac_data.xlsx', index_col='Fecha')
naftrac.columns = naftrac.columns.str.upper()
prices = yf.download(naftrac.columns.tolist(), start='2020-01-31', end='2022-07-30')
prices = pd.DataFrame(prices.Close).dropna(axis = 1)
pricesint = prices.loc[prices.index.intersection(naftrac.index.unique())]
pricesret = pricesint.pct_change().apply(lambda x: np.log(1+x)).dropna()

todropp = ['ALPEKA.MX', 'BSMXB.MX', 'GCC.MX', 'GENTERA.MX', 'IENOVA.MX', 'NEMAKA.MX', 'Q.MX', 'RA.MX', 'VESTA.MX', 'VOLARA.MX']
todropn = ['MXN.MX']