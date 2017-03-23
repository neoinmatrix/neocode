# coding=utf-8

# import sys
# print sys.path


import pandas
# print "hello world"
df = pandas.read_csv('sdata.csv', header='infer',  sep=',', \
    names=['user_id', 'item_id', 'rating', 'timestamp'])

df=df.drop(["timestamp"], axis=1)

df[['user_id', 'item_id']] -= 1
df['rating'] -= df['rating'].mean()

print df.head();
n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()

print n_users
print n_items