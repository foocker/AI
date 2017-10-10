# https://www.zhihu.com/question/38417101 理论参考
import numpy as np
from sklearn.datasets import load_boston

class PCAData(object):
    def zeroMean(self, dataMat):
            meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
            newData = dataMat - meanVal
            # print("标准差:", np.std(newData, axis=0))  # np.nanstd()
            normalizeData = newData / np.std(newData, axis=0)  # 归一化
            # print("归一化后的数据:", normalizeData)
            return newData, normalizeData, meanVal

    def percentage2n(self, eigVals, percentage):
        sortArray = np.sort(eigVals)  # 升序
        sortArray = sortArray[-1::-1]  # 逆转，即降序
        arraySum = sum(sortArray)
        tmpSum = 0
        num = 0
        for i in sortArray:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum * percentage:
                return num

    def pca(self, dataMat, percentage=0.99):
        newData, normalizeData, meanVal = self.zeroMean(dataMat)
        covMat = np.cov(normalizeData, rowvar=0)  # 求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        n = self.percentage2n(eigVals, percentage)  # 要达到percent的方差百分比，需要前n个特征向量
        print("选择的特征向量个数:", n)
        eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
        print("eigValIndice: ", eigValIndice)
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
        lowDDataMat = normalizeData * n_eigVect  # 低维特征空间的数据
        reconMat = (lowDDataMat * n_eigVect.T) + meanVal  # 重构数据
        return lowDDataMat, reconMat

    def run(self):
        np.random.seed(2)
        data_matrix = np.random.randint(25,size=(10,10)) # 生成范围在0-25的10*10的np-array
        data_matrix_choosed = data_matrix[:, :-1]        # 若最后一列为lables,则去掉lables
        print(data_matrix.shape)
        print("源数据:", data_matrix)
        lowDDataMat, reconMat = self.pca(data_matrix_choosed)
        print("重构数据:", reconMat)
        
        # boston = load_boston()    # 采用sklearn中boston的数据
        # X, y = boston['data'], boston['target']
        # print("boston_data:", X.shape)
        # lowDDataMat, reconMat = self.pca(X)    # 最终从13个特征中选择了12个特征
        # print("重构数据:", reconMat)

if __name__ == '__main__':
    charcatersobject = PCAData()
    charcatersobject.run()
