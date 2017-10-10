#!/usr/bin/env python
# coding=utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 构造数据:x_3*3
def g(x, y):
    return (x+y+x*y-0.3, y+1)

x, y = np.fromfunction(g,(3,3))
print('x={}'.format(x))

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet

print('normlize:',autoNorm(x))

def file_matrix(filename):
    f = open(filename)
    arraylines = f.readlines()
    numberoflines = len(arraylines)
    returnMat = np.zeros((numberoflines-1, 4))    # 去掉第一行(描述行)
    classLabelVector = []
    index = 0
    for line in arraylines[1:]:
        line = line.strip()
        listFromline = line.split(',')
        returnMat[index,:] = listFromline[0:4]
        classLabelVector.append(listFromline[-1])
        index += 1
    return [returnMat, classLabelVector]
result =file_matrix('iris.data') # 数据矩阵150*4,标签列
print("origina_data:", result[0],"rows".format(len(result[1])))

# 将实际数据放入归一化函数中
old_matrix = result[0]
new_matrix = autoNorm(old_matrix)
print("normalized_data", new_matrix[:10,])

# 作图看看
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(new_matrix[:,1],new_matrix[:,2])    # 作图按需作改动
plt.xlabel('colum1')
plt.ylabel('colum2')
plt.show()
