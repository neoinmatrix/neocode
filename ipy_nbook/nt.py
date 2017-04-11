# coding: utf-8
import pandas as pd
import numpy as np

train=pd.read_csv("small_train.csv")
def plotnumber(train,index):
    print train.iloc[index,0]
    img=train.iloc[index,1:].values.reshape(28,28)
    plt.imshow(img)
    plt.show()
plotnumber(train,11)
exit()
# normalize the train data
label_train=train['label']
train=train.drop('label', axis=1)
train = train / 255
# test = test / 255
train['label'] = label_train

from sklearn import decomposition

## PCA decomposition
pca = decomposition.PCA(n_components=50)
pca.fit(train.drop('label', axis=1))
PCtrain = pd.DataFrame(pca.transform(train.drop('label', axis=1)))
PCtrain['label'] = train['label']

from sklearn.neural_network import MLPClassifier
y = PCtrain['label'][0:1800]
X=PCtrain.drop('label', axis=1)[0:1800]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,), random_state=1)
clf.fit(X, y)

from sklearn import  metrics
#accuracy and confusion matrix
predicted = clf.predict(PCtrain.drop('label', axis=1)[1801:2000])
expected = PCtrain['label'][1801:2000]

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
