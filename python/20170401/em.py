#! /usr/bin/env python  
#coding=utf-8  
''''' 
author:zhaojiong 
EM算法初稿2016-4-28 
初始化三个一维的高斯分布， 
 
'''  
from numpy  import *  
import numpy as np  
import matplotlib.pyplot as plt  
import copy   

def init_em(x_num=2000):  
    global  mod_num,mod_prob_arr,x_prob_mat,theta_mat,theta_mat_temp,x_mat,mod_prob_arr_test  
    mod_num=3  
    x_mat =zeros((x_num,1))  
    mod_prob_arr=[0.3,0.4,0.3] #三个状态  
    mod_prob_arr_test=[0.3,0.3,0.4]  
      
      
    x_prob_mat=zeros((x_num,mod_num))  
    #theta_mat =zeros((mod_num,2))  
    theta_mat =array([ [30.0,4.0],  
                       [80.0,9.0],  
                       [180.0,3.0]  
                    ])  
    theta_mat_temp =array([ [20.0,3.0],  
                            [60.0,7.0],  
                            [80.0,2.0]  
                            ])  
    for i in range(x_num):  
        if np.random.random(1)<=mod_prob_arr[0]:  
            x_mat[i,0] = np.random.normal()*math.sqrt(theta_mat[0,1]) + theta_mat[0,0]  
        elif np.random.random(1)<= mod_prob_arr[0]+mod_prob_arr[1]:  
            x_mat[i,0] = np.random.normal()*math.sqrt(theta_mat[1,1]) + theta_mat[1,0]  
        else :   
              x_mat[i,0] = np.random.normal()*math.sqrt(theta_mat[2,1]) + theta_mat[2,0]  

    return x_mat        
def e_step(x_arr):  
    x_row ,x_colum =shape(x_arr)  
    global  mod_num,mod_prob_arr,x_prob_mat,theta_mat,theta_mat_temp,mod_prob_arr_test  
    for i in range(x_row):  
        Denom = 0.0  
        for j in range(mod_num):  
            exp_temp=math.exp((-1.0/(2*(float(theta_mat_temp[j,1]))))*(float(x_arr[i,0]-theta_mat_temp[j,0]))**2)  
              
            Denom += mod_prob_arr_test[j]*(1.0/math.sqrt(theta_mat_temp[j,1]))*exp_temp  
          
        for j in range(mod_num):  
            Numer = mod_prob_arr_test[j]*(1.0/math.sqrt(theta_mat_temp[j,1]))*math.exp((-1.0/(2*(float(theta_mat_temp[j,1]))))*(float(x_arr[i,0]-theta_mat_temp[j,0]))**2)  
            # if(Numer<1e-6):  
            #    Numer=0.0  
            if(Denom!=0):  
               x_prob_mat[i,j] = Numer/Denom  
            else:  
                x_prob_mat[i,j]=0.0  
    return x_prob_mat  
def m_step(x_arr):  
    x_row ,x_colum =shape(x_arr)  
    global  mod_num,mod_prob_arr,x_prob_mat,theta_mat,theta_mat_temp,mod_prob_arr_test  
    for j in range(mod_num):  
        MU_K = 0.0  
        Denom = 0.0  
        MD_K=0.0  
        for i in range(x_row):  
            MU_K += x_prob_mat[i,j]*x_arr[i,0]  
            Denom +=x_prob_mat[i,j]   
             
        theta_mat_temp[j,0] = MU_K / Denom   
        for i in range(x_row):  
            MD_K +=x_prob_mat[i,j]*((x_arr[i,0]-theta_mat_temp[j,0])**2)  
          
        theta_mat_temp[j,1] = MD_K / Denom  
        mod_prob_arr_test[j]=Denom/x_row  
          
      
    return theta_mat_temp  
def main_run(iter_num=500,Epsilon=0.0001,data_num=2000):  
    init_em(data_num)  
    global  mod_num,mod_prob_arr,x_prob_mat,theta_mat,theta_mat_temp,x_mat,mod_prob_arr_test  
    theta_row ,theta_colum =shape(theta_mat_temp)  
    # print "sdfsdf"
    for i in range(iter_num):  
        Old_theta_mat_temp=copy.deepcopy(theta_mat_temp)  
        x_prob_mat=e_step(x_mat)  
        theta_mat_temp= m_step(x_mat)  
        # print sum(abs(theta_mat_temp-Old_theta_mat_temp))
        if sum(abs(theta_mat_temp-Old_theta_mat_temp)) < Epsilon:  
           print "第 %d 次迭代退出" %i  
           break  
    return theta_mat_temp  
def plot_data(x_mat):  
    plt.hist(x_mat[:,0],200)  
    plt.show()  
def test(data_num):  
    testdata=init_em(data_num)  
    print testdata   
    #print '\n'  
    plot_data(testdata)  

print main_run()