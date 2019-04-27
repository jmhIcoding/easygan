#coding:utf-8
__author__ = 'dk'
import  json
import  random
from matplotlib import  pyplot as plt
num_max =1000
r=3
d=2*r
point=[]
delta= d/num_max
x=-r
point_x = []
point_y = []
while len(point) < 2*num_max:
    x=round(x,8)
    y=(r**2 - x**2)**0.5
    y= round(y,8)
    point.append([x,y])
    point.append([x,-y])
    point_x.append(x)
    point_y.append(y)
    point_x.append(x)
    point_y.append(-y)
    x+=delta
plt.scatter(point_x,point_y,linewidths=0.01)
#plt.show()

class data_generator:
    def __init__(self):
        self.index = 0
        self.epoch = 0
    def next_batch(self,batchsize):
        rst = []
        if (self.index+1)*batchsize >len(point):
            self.index = 0
            self.epoch +=1
        rst=point[self.index*batchsize:(self.index+1)*batchsize]
        self.index +=1
        return rst

if __name__ == '__main__':
    data_gen = data_generator()
    print(data_gen.next_batch(10))


