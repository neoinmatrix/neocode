# coding=utf-8
import numpy as np
import pandas as pd
import pylab as plt

from math import radians, cos, sin, asin, sqrt  
  
def haversine(lon1, lat1, lon2, lat2):
    # 将十进制度数转化为弧度  
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  
    # haversine公式  
    dlon = lon2 - lon1   
    dlat = lat2 - lat1   
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2  
    c = 2 * asin(sqrt(a))   
    r = 6371 # 地球平均半径，单位为公里  
    return c * r * 1000  

df=pd.read_csv("../data/440117.csv")
df=df[df["end_lat"]<32]
df['add_time'] = pd.to_datetime(df['add_time'],unit='s')
idxgt=df["add_time"]>"2018-01-21 11:00:02" 
idxlt=df["add_time"]<"2018-01-21 13:00:02"
df=df[idxgt][idxlt]
df=df.reset_index()
# print df.head()

# print haversine(113.5913554157962,23.55088174186293,113.5801542642146,23.55575248016203)
print haversine(113.58477216649469,23.545206628697514
    ,113.58405210766634,23.557168603385055)
exit()
import math
def bd09togcj02(bd_lon, bd_lat):
    """
    百度坐标系(BD-09)转火星坐标系(GCJ-02)
    百度——>谷歌、高德
    :param bd_lat:百度坐标纬度
    :param bd_lon:百度坐标经度
    :return:转换后的坐标列表形式
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * np.pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * np.pi)
    gg_lng = z * math.cos(theta)
    gg_lat = z * math.sin(theta)
    return [gg_lng, gg_lat]

for i in range(6):
    # a=bd09togcj02(df.loc[i,"start_lng"],df.loc[i,"start_lat"])
    a=[df.loc[i,"start_lng"],df.loc[i,"start_lat"]]
    # print df.loc[i,"start_lng"],df.loc[i,"start_lat"]
    # print a
    # b=bd09togcj02(df.loc[i,"end_lng"],df.loc[i,"end_lat"])
    b=[df.loc[i,"end_lng"],df.loc[i,"end_lat"]]
    dis=haversine(a[0],a[1],b[0],b[1])
    print df.loc[i,"length"],dis,df.loc[i,"length"]-dis

# print haversine(df.loc[1,"start_lng"],df.loc[1,"start_lat"],
#     df.loc[1,"end_lng"],df.loc[1,"end_lat"])
# print df.head()
# plt.ion()
# ann=plt.annotate('xxx',xy=(23.535,113.605))
# i=1

# for v in df.index:
#     a=[df.loc[v,"start_lat"],df.loc[v,"end_lat"]]
#     b=[df.loc[v,"start_lng"],df.loc[v,"end_lng"]]
#     plt.plot(a,b)
#     print df.loc[v,"add_time"]
#     plt.text(a[0], b[0],str(i) ,fontsize = 8,alpha=1)
#     i=i+1
#     ann.set_text(str(df.loc[v,"add_time"]))
#     # plt.text(23.535,113.605, str(df.loc[v,"add_time"]),fontsize = 16,alpha=1)
#     plt.scatter(df["end_lat"],df["end_lng"],c='r')
#     plt.scatter(df["start_lat"],df["start_lng"],c='b')
#     [(x1,y1)]=plt.ginput(1)
#     # plt.pause(1)
# plt.ioff()
# plt.show()