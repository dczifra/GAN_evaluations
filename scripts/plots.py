import matplotlib.pyplot as plt
import numpy as np
import sys

def different_model_compare( files,title,labels,
        xlabel="Minta meret",ylabel="[Párosítás]/[Minta meret]",second=False):

    colors=['r','g','b','y','p']
    fig, ax1 = plt.subplots()
    if(second): ax2=ax1.twinx()

    i=0
    for file in files:
        test=open(file)
        a,b,delta=np.array(test.readline().split(" ")).astype(np.int)
        data=np.array(test.readline().split(" "))[:-1].astype(np.float)
        if(second):
            data2=np.array(test.readline().split(" "))[:-1].astype(np.float)
            #print(data2)
        
        ax1.plot(range(a,b+delta,delta),data,colors[i],label=labels[i])
        if(second):
            ax2.plot(range(a,b+delta,delta),data2,colors[i]+'--',label=labels[i])
        i+=1
    
    plt.xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.title(title)
    ax1.legend()
    plt.show()

"""
def wgan():
    title="WGAN Párosítás-Pontszám "
    labels=["wgan 1000","wgan 5000","wgan 10000"]
    different_model_compare( files=["data/plot/train-wgan_compare_1000.txt",
            "data/plot/train-wgan_compare_5000.txt",
            "data/plot/train-wgan_compare_10000.txt"],
            title=title,
            labels=labels)

def wgan_wgan_gp(N):
    title="WGAN, WGAN-GP Párosítás-Pontszám {} EPOCH".format(N)
    labels=["test","wgan","wgan-gp","FID"]
    different_model_compare( files=["data/plot/train-test_compare.txt",
            "data/plot/train-wgan_compare_{}.txt".format(N),
            "data/plot/train-wgan-gp_compare_{}.txt".format(N)],
            title=title,
            labels=labels)
            """

def wgan_wgan_gp_type(N,type,second=False):
    title="WGAN, WGAN-GP Párosítás-Pontszám {} EPOCH".format(N)
    labels=["test","wgan","wgan-gp"]
    different_model_compare( files=["data/mnist/test/{}.txt".format(type),
            "models/wgan/generator_{}/{}.txt".format(N,type),
            "models/wgan-gp/generator_{}/{}.txt".format(N,type)],
            title=title,
            labels=labels,second=second)

def model_type(N,model,type,second=False):
    title="WGAN, WGAN-GP Párosítás-Pontszám {} EPOCH".format(N)
    labels=["test",model+" 1000",model+" 5000", model+" 10000"]
    different_model_compare( files=["data/mnist/test/{}.txt".format(type),
            "models/{}/generator_{}/{}.txt".format(model,1000,type),
            "models/{}/generator_{}/{}.txt".format(model,5000,type),
            "models/{}/generator_{}/{}.txt".format(model,10000,type)],
            title=title,
            labels=labels,second=second)

def FID_for_MNIST():
    title="FID for MINST"
    labels=["test","wgan","wgan-gp"]
    different_model_compare( files=[
            "eval/mnist_train_test.fid",
            "eval/mnist_train_wgan.fid",
            "eval/mnist_train_wgan-gp.fid"],
            title=title,
            labels=labels)
"""
def deficit():
    title="WGAN, WGAN-GP Deficit-Pontszám"
    labels=["wgan-gp 1000","wgan 1000","test"]
    different_model_compare( files=[
            "data/plot/train-wgan-gp_compare_1000.txtdeficit",
            "data/plot/train-wgan_compare_1000.txtdeficit",
            "data/plot/train-test_compare_.txtdeficit"],
            title=title,
            labels=labels,
            xlabel="Adathalmaz mérete",ylabel="FID score",r=[0,30])"""

#wgan_wgan_gp(1000)
#wgan_wgan_gp(5000)
#wgan_wgan_gp(10000)
#deficit()
#wgan()
if(__name__=="__main__"):
    #FID_for_MNIST()
    if(len(sys.argv)>1):
        N=int(sys.argv[1])
    else: N=1000

    model="wgan-gp"
    for type in ["compare","flow","deficit","defFlow","fid_score"]:
        model_type(N,model,type,type=="defFlow")
        #wgan_wgan_gp_type(N,type,type=="defFlow")
    exit()
