# coding=utf-8
import numpy as np
import pandas 
import random
import math

from numpy.linalg import norm
from pandas import DataFrame
from numpy.random import normal
import pandas 

def loss(R, U, V):
    E = 0.0
    for (t, i, j, rij) in R.itertuples():
        E += (rij - np.dot(U[i], V[j]))**2  + norm(U[i]) + norm(V[j])
    return E 

if __name__ == '__main__':
    latent_dimensions = 200
    alpha = 0.1
    eta0 = 0.01
    epochs = 5
    power_t = 0.25

    df = pandas.read_csv('sdata.csv', header='infer',  sep=',', 
         names=['user_id', 'item_id', 'rating', 'timestamp'])
    df=df.drop(["timestamp"], axis=1)
    df[['user_id', 'item_id']] -= 1
    # df['rating'] -= df['rating'].mean()

    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    # U=DataFrame()
    # V=DataFrame()
    # U = DataFrame(normal(size=(latent_dimensions, n_users)),
    #               columns=df['user_id'].unique())
    # # print U.head()
    # V = DataFrame(normal(size=(latent_dimensions, n_items)),
    #               columns=df['item_id'].unique())
    U=pandas.DataFrame.from_csv("./U.csv")
    U.columns=df['user_id'].unique()
    # print V.head()
    V=pandas.DataFrame.from_csv("./V.csv",header=0)
    V.columns=df['item_id'].unique()
    # print V.head()
    # df.ix[0:19][""].values
    items=df.ix[0:19]["item_id"].values
    # print V[items]
    # print U[0]
    # print V[30]
    # exit()
    # print type(items)
    # exit()
    i=0
    mean=df["rating"].mean()
    for item_id in items:
        print item_id," ",int(np.dot(V[item_id],U[0])+mean)," ",df["rating"][i]
        i+=1
        # break 

    # print loss(df,U,V)

    # 4519904.18297
    # 193898.760532