#coding:utf-8

import numpy as np
import gzip #这似乎是python自带的库，无需pip安装
#from cv2 import imwrite #用来在硬盘上存下图片
#（似乎存图没有很大意义。不使用了）

#以下抽取图片和标签的程序最初来自
#https://blog.csdn.net/juanjuan1314/article/details/77979582 

# SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

#按32位读取，主要为读校验码、图片数量、尺寸准备的
#仿照tensorflow的mnist.py写的。
def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

#抽取图片，并按照需求，可将图片中的灰度值二值化，按照需求，可将二值化后的数据存成矩阵或者张量
#仿照tensorflow中mnist.py写的
def extract_images(input_file, is_value_binary=True, is_matrix=True):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic !=2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %(magic, input_file.name))
        num_images = _read32(zipf)
        rows = _read32(zipf)
        cols = _read32(zipf)
        print('magic',magic,'图片数量:',num_images, '行数:',rows, '列数:',cols)
        buf = zipf.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        if is_matrix:
            data = data.reshape(num_images, rows*cols)
        else:
            data = data.reshape(num_images, rows, cols)
        if is_value_binary:
            return [np.minimum(data, 1),num_images,rows*cols]
        else:
            return [data,num_images,rows*cols,rows*cols]

#抽取标签
#仿照tensorflow中mnist.py写的
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return [labels,num_items]

#以上抽取图片和标签的程序最初来自
#https://blog.csdn.net/juanjuan1314/article/details/77979582 

if __name__ == '__main__':
    [images,numImages,imgSize]=extract_images(TRAIN_IMAGES)
    [labels,numLabels]=extract_labels(TRAIN_LABELS)
    
    [testImages,numTestImages,testImgSize]=extract_images(TEST_IMAGES)
    [testLabels,numTestLabels]=extract_labels(TEST_LABELS)
    
    if(numImages==numLabels and numTestImages==numTestLabels and imgSize==testImgSize):
    #如果两个数据集中图片数量等于标签数量，且图片尺寸都相同
        #下面开始求每一张测试图是哪个数字
        #原理：求测试图“最像”数据库里的那哪些图片
        #如果测试图与数据库中某张图片的某个像素点不同，则测试图到这张图片的距离+1
        #选出数据集中离测试图距离最小的若干张图片（"若干"==neighborCount）
        neighborCount=5 #要选出数据库中前多少张与测试图最像的图片
        indexList=[0]*neighborCount #记录数据库中离测试图距离最近的neighborCount张图的下标。 
        distList=[imgSize]*neighborCount  #记录一张测试图与数据库中各图的前neighborCount个最短距离
        maxIndex=0#记录distList里最大dist所在的下标

        correct=0#记录正确个数
        tests=numTestImages    #要拿测试图的前多少张进行验证
        #tests=10
        samples=numImages    #要拿数据库中前多少张来作为样本
        #samples=5000
        
        images=images[0:samples]  #取前samples个样本 #不要写samples-1！
        for i in range(0,tests):    #验证前tests张测试图
            testCopy=np.tile(testImages[i],(samples, 1)) #这样测试图一共有samples份
            #np.tile(matrix, (2, 1))#沿X轴复制1倍（相当于没有复制），再沿Y轴复制2倍
            testCopy=np.bitwise_xor(testCopy,images) #测试图每个像素点与数据库每个像素点比较的结果
            testCopy=np.sum(testCopy,1) #得到测试图与数据库中每张图的距离
            
            #下面把所有的下标按距离从小到大排序。
            argSorted=np.argsort(testCopy)
            
            #下面由距离最短的neighborCount个距离对识别结果投票
            vote=[0]*10 #记录投票结果
            for k in range(0,neighborCount):
                vote[labels[argSorted[k]]]+=1#由数据库的index查询数字，给数字投票
            maxVote=0   #记录最大得票数
            maxVoteIndex=10 #记录最大得票数的下标（即认为测试图是什么数字）
            for k in range(0,10):
                if(maxVote<vote[k]):
                    maxVote=vote[k]
                    maxVoteIndex=k
            #认为maxVoteIndex就是测试图i的识别结果
            if(maxVoteIndex==testLabels[i]):#如果识别正确
                correct+=1  #正确个数+1
            if(i%100==99):
                print("已识别",i+1,"张测试图。目前正确率(%):",correct/(i+1)*100)
        print("最终正确率(%):",correct/tests*100)
