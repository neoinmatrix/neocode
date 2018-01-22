import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("../data/440117.csv")
# print df.head()
df=df[df["end_lat"]<32]
df['add_time'] = pd.to_datetime(df['add_time'],unit='s')
# print df.head()
idxgt=df["add_time"]>"2018-01-21 11:00:02" 
idxlt=df["add_time"]<"2018-01-21 13:00:02"
df=df[idxgt][idxlt]
# print dfx.describe()
# print df["end_lat"]>32
# exit()
# df= df.drop()
plt.ion()
for v in df.index:
    a=[df.loc[v,"start_lat"],df.loc[v,"end_lat"]]
    b=[df.loc[v,"start_lng"],df.loc[v,"end_lng"]]
    plt.plot(a,b)

    # print v
    plt.scatter(df["end_lat"],df["end_lng"],c='r')
    plt.scatter(df["start_lat"],df["start_lng"],c='b')
    plt.pause(0.1)
plt.ioff()
plt.show()
# print df["length"].mean()
# print df["length"].min()
# print df["length"].max()
# print df["length"].std()