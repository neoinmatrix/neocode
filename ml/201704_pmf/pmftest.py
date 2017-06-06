# coding=
import numpy as np
import pandas 
import random
import math

import ipyparallel as ipp
from numpy.linalg import norm
from pandas import DataFrame
from numpy.random import normal

def loss(R, U, V):
    E = 0.0
    for (t, i, j, Rij) in R.itertuples():
    # for iter in range(R.index): 
        E += (Rij - np.dot(U[i], V[j]))**2  + norm(U[i]) + norm(V[j])
    return E

def mf(df,n_users,n_items,latent_dims=10,lambda_u=0.1,lambda_v=0.1, learn_rate=0.01, iters=3,converge=1e-5):
    U = DataFrame(normal(size=(latent_dims, n_users)), columns=[i for i in range(n_users)])
    V = DataFrame(normal(size=(latent_dims, n_items)), columns=[i for i in range(n_items)])
    t = 1.0
    index = df.index.values
    likelihood=0.0
    likelihood_old=1.0
    # import numpy as np
    # f=open('d_'+str(os.getpid()),'w')
    # f.write("hesdfsdf")
    # f.close()
    # return U,V

    # random.shuffle(index)
    for iterx in xrange(iters):
        random.shuffle(index)
        for count, pos in enumerate(index):
            i, j, Rij = df.ix[pos] 
            Ruv = np.dot(U[i], V[j])
            U[i] = U[i] - learn_rate * ((Ruv - Rij) * V[j] + lambda_u * U[i])
            V[j] = V[j] - learn_rate * ((Ruv - Rij) * U[i] + lambda_v * V[j])
            likelihood+= 0.5*np.dot(U[i],U[i])*lambda_u
            likelihood+= 0.5*np.dot(V[i],V[i])*lambda_v
            if np.isnan(U.values).any() or np.isnan(V.values).any():
                raise ValueError('overflow')
            t += 1
            if int(t)%2000==0:
                break
        tmpx="iterx %d %f %f %f"%(iterx,likelihood,likelihood_old,(np.fabs(likelihood-likelihood_old)/likelihood_old))
        print tmpx

        # f=open('d_'+str(os.getpid()),'w')
        # f.write(tmpx)
        # f.close()
        if (np.fabs(likelihood-likelihood_old)/likelihood_old)<converge:
            print np.fabs(likelihood-likelihood_old)
            break
        else:
            likelihood_old=likelihood
            likelihood=0.0
    return U, V

def fit(df,n_users,n_items,latent_dims=10,lambda_u=0.1,lambda_v=0.1, learn_rate=0.01, iters=3,converge=1e-5):
    rc=ipp.Client()
    dview = rc[:]
    k = float(len(rc))
    with dview.sync_imports():
        import random 
        from numpy.linalg import norm
        from numpy import dot, isnan
        from numpy.random import normal
        from pandas import DataFrame
        import os
        # import numpy as np

    dview.scatter('df', df)
    res = dview.apply_sync(mf, n_users,n_items,latent_dims=10,lambda_u=0.1,lambda_v=0.1, learn_rate=0.01, iters=3,converge=1e-5)
    add = lambda a,b: a.add(b, fill_value=0)
    U = reduce(add, (r[0] for r in res))/k
    V = reduce(add, (r[1] for r in res))/k
    return U, V

def predict(df,U,V):
    pass

if __name__ == '__main__':
    latent_dims = 20       # the latent factor dimensions
    lambda_u=0.1            # the rate for U
    lambda_v=0.1            # the rate for V
    learn_rate = 0.01       # the learning in gradient descent
    iters = 50             # the iterations in learning
    converge=1e-3           # the converge when smaller then stop
    
    df = pandas.read_csv('n_user_train.df', header='infer', names=['user_id', 'item_id', 'rating'])
    # df[['user_id', 'item_id']] -= 1
    df['rating'] -= df['rating'].mean()
    # print df.index
    # 1189616.27619

    # exit()
    # users:1769
    # items:21493
    # line:39133
    n_users = 1769
    n_items = 21493
    U = DataFrame(normal(size=(latent_dims, n_users)), columns=[i for i in range(n_users)])
    V = DataFrame(normal(size=(latent_dims, n_items)), columns=[i for i in range(n_items)])
    import time
    start = time.clock()
    # print loss(df, U, V)
    # U, V = mf(df, latent_dims=latent_dims, lambda_u=lambda_u,lambda_v=lambda_v, learn_rate=learn_rate, iters=iters)
    U, V = mf(df,n_users,n_items, latent_dims=latent_dims, lambda_u=lambda_u,lambda_v=lambda_v, learn_rate=learn_rate, iters=iters,converge=converge)
    end = time.clock()
    print "read: %f s" % (end - start)
    
    print loss(df, U, V)
    U.to_csv("./U.csv")
    V.to_csv("./V.csv")
   