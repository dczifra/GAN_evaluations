import numpy as np
import os
import eval.FID
from keras.datasets import mnist

def calc_fid(train,test,out="FID_mnist_train_test.txt"):
    myfile=open(out,"w")
    myfile.write("100 2000 100\n")
    N=2000
    r=100

    myrange=[]
    for i in range(r,r+N,r):
        np.random.shuffle(train)
        np.random.shuffle(test)
        res=eval.FID.FID(train[i:i+r],test[i:i+r])
        print(res)
        myrange.append(res)
        myfile.write(str(res)+" ")

    myfile.close()
    import matplotlib.pyplot as plt
    plt.plot(range(r,N+r,r),myrange)


def read_dataset(path):
    print(path)
    dataset=[]
    for file in os.listdir(path):
        mystream=open(path+file)
        img=[]
        for row in mystream:
            img.append(row.split(" ")[:-1])
        dataset.append(img)
        mystream.close()
    
    size=np.shape(dataset)
    print(size)
    dataset=np.resize(dataset,(size[0],1,size[1],size[2]))
    dataset=np.repeat(dataset,3,1)
    np.random.shuffle(dataset)
    print(np.shape(dataset))
    return dataset

def fid_score(train_file,test_file):
    train=read_dataset(train_file+"/data/")
    test=read_dataset(test_file+"/data/")
    print(np.shape(train),np.shape(test),test_file)
    calc_fid(train,test,test_file+"/fidscore.txt")
    

if(__name__=="__main__"):
    1
    #train=read_dataset("data/mnist/train/data")
    #test=read_dataset("data/mnist/wgan_1000/")
    #calc_fid(train,test,"eval/mnist_train_wgan.fid")
