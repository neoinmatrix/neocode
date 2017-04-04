# coding=utf-8
import numpy as np
import pandas as pd
import random
import math

from numpy.linalg import norm
from pandas import DataFrame
from numpy.random import normal

def loss(R, U, V):
    E = 0.0
    for (t, i, j, rij) in R.itertuples():
        E += (rij - np.dot(U[i], V[j]))**2  + norm(U[i]) + norm(V[j])
    return E 

def ctr(rt,cp=None, params=None):
    """ stochastic gradient descent """
    n_users = rt['user_id'].nunique()
    n_items = rt['item_id'].nunique()
    latent_dims=params["latent_dims"]
    n_users=params["n_users"]
    n_items=params["n_items"]

    U = DataFrame(normal(size=(latent_dims, n_users)),
                  columns=rt['user_id'].unique())
    V = DataFrame(normal(size=(latent_dims, n_items)),
                  columns=rt['item_id'].unique())
    
    for epoch in xrange(epochs):
        for count, pos in enumerate(index):
            pass
            # i, j, rij = rt.ix[pos] 
            # eta =  eta0 / (t ** power_t)

            # rhat = np.dot(U[i], V[j])

            # U[i] = U[i] - eta * ((rhat - rij) * V[j] + alpha * U[i])
            # V[j] = V[j] - eta * ((rhat - rij) * U[i] + alpha * V[j])
            # # print ((rhat - rij) * V[j] + 64 * U[i])
            # # exit()
            # if np.isnan(U.values).any() or np.isnan(V.values).any():
            #     raise ValueError('overflow')
            
            # t += 1
    return U, V

if __name__ == '__main__':
    params={
        "n_users":100,
        "n_items":1000,
        "latent_dims":200,

        "lambda_u":0.01,
        "lambda_v":100,

        "max_iter":200,
        "a":1.0,
        "b":0.01,
    }

    print params
    exit() 

    rt = pd.read_csv('rate.csv', header='infer', sep=',', 
         names=['user_id', 'item_id', 'rating'])
    # rt=rt.drop(["timestamp"], axis=1)
    rt[['user_id', 'item_id']] -= 1
    rt['rating'] -= rt['rating'].mean()

    n_users = rt['user_id'].nunique()
    n_items = rt['item_id'].nunique()

    params["n_users"]=n_users
    params["n_items"]=n_items

    U = pd.DataFrame(normal(size=(latent_dims, n_users)),
                  columns=rt['user_id'].unique())
    # print U.info()

    V = pd.DataFrame(normal(size=(latent_dims, n_items)),
                  columns=rt['item_id'].unique())
    # print U[0]

    # print normal(size=(latent_dimensions, n_items))
    # print loss(rt, U, V)
    # exit()
    # U, V = fit(rt, alpha=alpha, eta0=eta0, power_t=power_t, 
    #            epochs=epochs, latent_dimensions=latent_dimensions)

    U, V = ctr(rt, params,params)
    # print loss(rt, U, V)
    U.to_csv("./U.csv")
    V.to_csv("./V.csv")
   