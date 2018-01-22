# coding=utf-8
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt
im = np.array(Image.open('map.jpg'))
plt.imshow(im)
a=plt.ginput(2)
print a

# # coding=utf-8
# from PIL import Image  
# from pylab import *  
  
# im = array(Image.open('map.jpg'))  
# imshow(im)  
# print 'Please click 3 points'  
# x =ginput(3)  
# print 'you clicked:',x  
# # show()  