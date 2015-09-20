#coding=utf-8
from numpy import *

def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        #map存储的是float类型的值
        fltLine = map(float, curLine)
        dataMat.append(fltLine)
    
    return dataMat

#计算距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

#为给定数据集构建一个包含k个随机质心的集合
def randCent(dataSet, k):
    #数据点的个数
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        #第j列的最小值
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k,1)

    return centroids

def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    #数据点的个数
    m = shape(dataSet)[0]
    #存储每个点的簇分配结果，m行，2列的矩阵
    #第一列为所属的类别，第二列为到所属类别的距离
    clusterAssment = mat(zeros((m,2)))
    #随机初始化k个质心；
    centroids = createCent(dataSet, k)

    clusterChanged = True
    #如果所有数据点的簇分配不再变化则结束
    while clusterChanged:
        clusterChanged = False;
        #对于每个数据点
        for i in range(m):
            minDist = inf; minIndex = -1
            #对于每个数据点，计算数据点和质心之间的距离，找到距离最近的质心
            for j in range(k):
                distJI = distMeans(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            #clusterAssment[i,0]表示第i行，第0列数据
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            #clusterAssment[i,:]表示第i行的数据
            clusterAssment[i,:] = minIndex, minDist**2
        
        #print clusterAssment

        #更新质心的位置
        for cent in range(1):
            #通过数组过滤来获得给定簇的所有点；
            #clusterAssment[:,0].A表示获取所有行的第一个数据，然后与cent比较，如果等于cent则对应的结果为1，取nonzero，得到相应的不等于0的列号
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            #计算所有点的均值，axis=0表示沿矩阵的列方向进行均值计算
            centroids[cent,:] = mean(ptsInClust, axis=0)

    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeans=distEclud):
    m = shape(dataSet)[0]
    #创建一个矩阵来存储数据集中每个点的簇分配结果以及平方误差
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    #保存所有质心
    centList = [centroid0]

    #遍历数据集中所有点计算每个点到质心的误差值
    for j in range(m):
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j,:]) ** 2


    while(len(centList) < k):
        lowestSSE = inf
        #遍历centList中的每一个簇
        for i in range(len(centList)):
            #尝试划分每一个簇
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            #划分后的误差计算
            sseSplit = sum(splitClustAss[:,1])
            #划分前的误差计算
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A != i)[0],1])

            print "sseSplit, and notSplit: ", sseSplit, sseNotSplit

            #如果划分前后的误差值之和小于lowestSSE，则本次划分将保存
            if(sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        #使用两个数组过滤器，更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit

        print 'the bestCentToSplit is: ', bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:] = bestClustAss

    return centList, clusterAssment


if __name__ == '__main__':
    dataMat = mat(loadDataSet('testSet.txt'))
    myCentroids, clustAssing = kMeans(dataMat, 4)
    dataMat1 = mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(dataMat1,3)
    print centList

