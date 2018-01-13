import numpy as np
import time
import multiprocessing

train=np.random.randn(10,10)
test=np.random.randn(5,10)

pca_num=10
pca = PCA(n_components=pca_num)
pca.fit(train)
train=pca.transform(train)
test=pca.transform(test)

print train
print test