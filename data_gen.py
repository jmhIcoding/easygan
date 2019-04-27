#using:utf-8
__author__ = 'dk'
import  json
num_max =5000
r=3
d=2*r
point=[]
delta= d/num_max
x=-r
while len(point) < 2*num_max:
    x=round(x,8)
    y=(r**2 - x**2)**0.5
    y= round(y,8)
    point.append([x,y])
    point.append([x,-y])
    x+=delta
print(len(point))
print(point)
