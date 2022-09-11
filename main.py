#%% IMPORTAR LIBRERIAS Y FUNCIONES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")
'''
Funciones alamacenadas en el archivo functions.py
'''
from functions import rand_weights
from functions import random_portfolio
from functions import summary_portfolios
from functions import best_portfolio

#%% VARIABLES INICIALES (NAFTRAC Y PRECIOS YF)
from data import comision
from data import capital
from data import naftrac
from data import prices
from data import pricesint
from data import pricesret
from data import todropn, todropp
from data import df_activa

#%% PLOT PCT_CHANGE DE RENDIMIENTOS
fig, ax = plt.subplots(figsize=(9,5))
sns.lineplot(data=prices.pct_change().dropna(), legend=False).set(title='Prices returns')

#%% CORR DE RENDIMIENTOS
fig, ax = plt.subplots(figsize=(6,6))
sns.heatmap(data=prices.pct_change().dropna().apply(lambda x: np.log(1+x)).corr(), cbar=False, cmap='Blues').set(title='Returns correlation')

#%% PORTAFOLIO DE PESOS EFICIENTES Y LOOP DE CONDICIONES
weights = pd.DataFrame(columns=['Means', 'StanDev', 'SharpeR', 'AC.MX', 'ALFAA.MX', 'ALPEKA.MX',
       'ALSEA.MX', 'AMXL.MX', 'ASURB.MX', 'BBAJIOO.MX', 'BIMBOA.MX',
       'BOLSAA.MX', 'BSMXB.MX', 'CEMEXCPO.MX', 'CUERVO.MX', 'ELEKTRA.MX',
       'FEMSAUBD.MX', 'GAPB.MX', 'GCARSOA1.MX', 'GCC.MX', 'GENTERA.MX',
       'GFINBURO.MX', 'GFNORTEO.MX', 'GMEXICOB.MX', 'GRUMAB.MX',
       'IENOVA.MX', 'KIMBERA.MX', 'KOFUBL.MX', 'LABB.MX', 'LIVEPOLC-1.MX',
       'MEGACPO.MX', 'NEMAKA.MX', 'OMAB.MX', 'ORBIA.MX', 'PE&OLES.MX',
       'PINFRA.MX', 'Q.MX', 'RA.MX', 'TLEVISACPO.MX', 'VESTA.MX',
       'VOLARA.MX', 'WALMEX.MX'], index=naftrac.index[12:31].tolist())

for i in range(12,31):
    '''
    Aquí estoy sacando portafolios eficientes sobre todos los meses entre 01/2022 y 07/2022.
    '''
    portfolio = summary_portfolios(5000, pricesret.iloc[:12])
    weights.iloc[i-12] = best_portfolio(portfolio).iloc[0]
    
weights = weights.iloc[:,3:]

for i in range(len(pricesret.iloc[:12])):
    for j in range(len(pricesret.columns)):
        change = pricesret.iloc[i+1,j]/pricesret.iloc[i,j]-1
        if change > .05:
            weights.iloc[i+1,j] = weights.iloc[i+1,j]*(1+.035)
        elif change < -.05:
            weights.iloc[i+1,j] = weights.iloc[i+1,j]*(1)

weights = ((weights*capital)/pricesint.iloc[12:31]).astype('int')
weights = weights*pricesint.iloc[12:31]
weights = weights.sum(axis=1)

df_activa['Capital'] = round(weights,2)
df_activa['Rend'] = round(weights.pct_change().fillna(0),4)
df_activa['Rend_acum'] = round(weights.pct_change().fillna(0),4).cumsum()
df_activa

#%%
fig, ax = plt.subplots(figsize=(7,5))
sns.histplot(naftrac.loc[:, (naftrac != 0).all(axis=0)].values, legend=False)

#%%naftrac = naftrac.replace(0, np.nan).dropna(axis=1)
naftracprices = prices.loc[prices.index.intersection(naftrac.index)]
naftracprices = naftracprices.drop(todropp, axis=1)
naftrac = naftrac.drop(todropn, axis=1)
capital = round(naftrac*capital/naftracprices,0)*naftracprices
capital = capital.sum(axis = 1)
df_pasiva = pd.DataFrame()
df_pasiva['Capital'] = round(capital,2)
df_pasiva['Rend'] = round(capital.pct_change().fillna(0),4)
df_pasiva['Rend_acum'] = round(capital.pct_change().fillna(0),4).cumsum()
df_pasiva.head(10)
# %%
