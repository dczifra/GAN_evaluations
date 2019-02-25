import numpy as np
import os
import FID
from keras.datasets import mnist

def calc_fid(train,test,out="FID_mnist_train_test.txt"):
    myfile=open(out,"w")
    myfile.write("100 1000 100\n")
    N=2000
    r=100

    myrange=[]
    for i in range(r,r+N,r):
        np.random.shuffle(train)
        np.random.shuffle(test)
        res=FID.FID(train[i:i+r+1],test[i:i+r+1])
        print(res)
        myrange.append(res)
        myfile.write(str(res)+" ")

    
    import matplotlib.pyplot as plt
    plt.plot(range(r,N+r,r),myrange)


def read_dataset(path):
    dataset=[]
    for file in os.listdir(path):
        mystream=open(path+file)
        img=[]
        for row in mystream:
            img.append(row.split(" ")[:-1])
        dataset.append(img)
        mystream.close()
    
    size=np.shape(dataset)
    dataset=np.resize(dataset,(-1,1,size[1],size[2]))
    dataset=np.repeat(dataset,3,1)
    np.random.shuffle(dataset)
    print(np.shape(dataset))
    return dataset
    

#mnist()
train=read_dataset("data/mnist/train/")
test=read_dataset("data/mnist/wgan_1000/")
calc_fid(train,test,"eval/mnist_train_wgan.fid")