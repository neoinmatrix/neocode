# coding=utf-8
import pandas as pd

train=pd.read_csv("./small_train.csv")
test=pd.read_csv("./small_test.csv")

print train
print format(train.shape)
