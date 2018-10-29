#coding:utf-8

import numpy as np
import gzip
from sklearn import tree

TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

def iris():    #教学视频里的例子
    from sklearn.datasets import load_iris  #教学视频里的例子
    iris=load_iris()
    test_idx=[0,50,100]
     
    train_target=np.delete(iris.target,test_idx)
    train_data=np.delete(iris.data,test_idx,axis=0)
     
    test_target=iris.target[test_idx]
    test_data=iris.data[test_idx]
     
    clf=tree.DecisionTreeClassifier()
    clf.fit(train_data,train_target)
    print(test_target)
    print(clf.predict(test_data))
    
    from sklearn.externals.six import StringIO
    import pydot
    dot_data=StringIO()
    tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True,rounded=True,
                         impurity=False)
    graph=pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("iris.pdf")

def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

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
        
def extract_labels(input_file):
    with gzip.open(input_file, 'rb') as zipf:
        magic = _read32(zipf)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, input_file.name))
        num_items = _read32(zipf)
        buf = zipf.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return [labels,num_items]
    
def entropy(predict):
    #香农熵。predict是正确答案的概率
    #H(actual,predict)=−∑actual(xi)log(predict(xi))
    return -predict*np.log2(predict)

if __name__ == '__main__':
    [images,numImages,imgSize]=extract_images(TRAIN_IMAGES)
    [labels,numLabels]=extract_labels(TRAIN_LABELS)
    [testImages,numTestImages,testImgSize]=extract_images(TEST_IMAGES)
    [testLabels,numTestLabels]=extract_labels(TEST_LABELS)
    
    if(numImages==numLabels and numTestImages==numTestLabels and imgSize==testImgSize):
    #如果两个数据集中图片数量等于标签数量，且图片尺寸都相同
        clf=tree.DecisionTreeClassifier()
        clf.fit(images,labels)
        testResult=clf.predict(testImages)
        
        correct=0
        for i in range(0,numTestLabels):
            if(testResult[i]==testLabels[i]):
                correct+=1
        print("正确率(%):",correct/numTestLabels*100)
