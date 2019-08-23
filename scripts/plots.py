import matplotlib.pyplot as plt
import numpy as np
import sys
import os


    
def different_model_compare( files,title,labels,
                             xlabel="Minta meret",ylabel="[Párosítás]/[Minta meret]",second=False, time=False, outfile="."):
    datas = []
    for file in files:
        test=open(file)
        a,b,delta=np.array(test.readline().split(" ")).astype(np.int)
        data=np.array(test.readline().split(" "))[:-1].astype(np.float)
        datas.append(([a,b,delta],data))
    plot_data(datas, title, labels, xlabel, ylabel, second, time, outfile)
    
def plot_data(datas, title, labels, xlabel, ylabel, second, time, outfile):

    colors=['r','g','b','y','p','r--']
    fig, ax1 = plt.subplots()
    if(second): 
        ax2=ax1.twinx()
        ax2.set_ylabel("Value of the flow/Max matching")

    i=0
    for ranges, data in datas:
        a,b,delta=ranges
        
        if(second):
            data2=np.array(test.readline().split(" "))[:-1].astype(np.float)
            print(data2)
            ax1.bar(range(a,b+delta,delta),data2,colors[i],label=labels[i])

        if(time and i==len(files)-1):
            ax2=ax1.twinx()
            ax2.set_ylabel("Value of the flow/Max matching")
            plt.plot(range(a,b+delta,delta),data)
        else:
            ax1.plot(range(a,b+delta,delta),data,colors[i],label=labels[i])
        i+=1

    plt.style.use('ggplot')
    plt.xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.title(title)
    ax1.legend()
    plt.savefig(outfile)
    plt.show()

def gen_data_progress_of_model(models, file, batch):

    outfiles = []
    for mod in models:
        outfile = "/".join(mod[0].split('/')[:-1])
        outfile = os.path.join(outfile, "epochs.txt")
        outfiles.append(outfile)
        outfile = open(outfile, "w")
        outfile.write("{} {} {}\n".format(1000, 1000*len(mod),1000))
        for epoch in mod:
            test=open(os.path.join(epoch,file))
            a,b,delta=np.array(test.readline().split(" ")).astype(np.int)
            data=np.array(test.readline().split(" "))[:-1].astype(np.float)
            outfile.write(str(data[batch])+" ")
        outfile.close()

    print(outfiles)
    return outfiles

from PIL import Image
def plotImages2(data, n_x, n_y, name, text=None):
    data = np.array(data)
    (height, width, channel) = data.shape[1:]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    if channel == 1:
        mode = "L"
        data = data[:,:,:,0]
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1), dtype='uint8')
    else:
        mode = "RGB"
        image_data = 50 * np.ones((height_inc * n_y + 1, width_inc * n_x - 1, channel), dtype='uint8')
    for idx in range(n):
        x = idx % n_x
        y = idx // n_x
        sample = data[idx]
        image_data[height_inc*y:height_inc*y+height, width_inc*x:width_inc*x+width] = 255*sample.clip(0, 0.99999)
    img = Image.fromarray(image_data,mode=mode)

    fileName = name + ".png"
    print("Creating file " + fileName)
    if text is not None:
        img.text(10, 10, text)
    img.save(fileName)
    
def read_model_data(model_path,N = -1 ,mod2="image"):
    model_data=[]
    i=-1
    while(i<N or N == -1):
        i+=1
        # Read the file:
        try:
            myfile=open(model_path+"/{}_{}.txt".format(mod2,i))
        except IOError:
            break;

        mtx=[]
        for line in myfile:
            mtx.append(line[:-2].split(' '))
        mtx=[[float(e) for e in row] for row in mtx]
        model_data.append(mtx)
    return model_data

def read_mathcing(mathing_file):
    myfile=open(mathing_file)
    match=[]

    for line in myfile:
        a,b,cost=line[:-1].split(' ')
        a,b=int(a),int(b)
        match.append([a,b,cost])
    
    return match

def plotImages(data, n_x, n_y, name, text=None):
    plt.close('all')
    plt.style.use('ggplot')

    data = np.array(data)
    print(np.shape(data))
    #data = color_resize(data)
    
    (height, width) = data.shape[1:3]
    height_inc = height + 1
    width_inc = width + 1
    n = len(data)
    if n > n_x*n_y: n = n_x * n_y

    plt.figure(figsize=(n_x,n_y))
    for i in range(n_x):
        for j in range(n_y):
            ax=plt.subplot(n_x,n_y,i*n_y+j+1)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
                    
            plt.imshow(np.resize(data[i*n_y+j], (height, width)))
            
    fileName = name + ".png"
    print("Creating file " + fileName)
    if text is not None:
        plt.title( text)
    plt.savefig(fileName)

def estimate_radius(pict1):
    size = np.prod(np.shape(pict1)[1:])
    pict1 = np.resize(pict1, (len(pict1),size))
    ret = []
    for pict in pict1:
        num = np.sqrt(np.sum(pict)/(np.pi*255))
        num = int(num)
        ret.append(num)
    return ret

def print_pictures(pict1, pict2, score,ind, outdir):
    plt.close('all')
    plt.style.use('ggplot')
    est1 = estimate_radius(pict1)
    est2 = estimate_radius(pict2)
    
    N=len(score)
    size=np.shape(pict1)[1:]
    #print(size, np.prod(size))
    plt.figure(figsize=(N,2))
    for i in range(N):
        vect=np.array(pict1[i])-np.array(pict2[i])
        vect=np.reshape(vect,(np.prod(size)))
        #vect=np.reshape(vect,(28*28))
        #print(i,np.linalg.norm(vect,2))
        ax=plt.subplot(2,N,i+1)

        plt.imshow(pict1[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title(score[i])
        
        ax=plt.subplot(2,N,N+i+1)
        plt.imshow(pict2[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        #ax.set_title(str(est1[i])+" "+str(est2[i]))
    plt.savefig('{}/matching{}.png'.format(outdir, ind))
    plt.show()

def get_nth_matching(model_path1, model_path2, matching_file):
    match=read_mathcing(matching_file)
    N=len(match)
    model_data1=read_model_data(model_path1, N)
    model_data2=read_model_data(model_path2, N)

    model_data1 = color_resize(model_data1)
    model_data2 = color_resize(model_data2)

    scores=[(float(match[i][2]),i) for i in range(len(match))]
    scores.sort()
    scores=scores[::N//100]
    pict1=[model_data1[match[int(i)][0]] for _,i in scores]
    pict2=[model_data2[match[int(i)][1]] for _,i in scores]
    #size=np.shape(model_data1)
    #if(size[-1]!=size[-2]):
    #    #print(size)
    #    model_data1=np.resize(model_data1,(size[0],size[1],size[2]//3,3)).astype(int)
    #    model_data2=np.resize(model_data2,(size[0],size[1],size[2]//3,3)).astype(int)
    #else:
    #    model_data1=np.resize(model_data1,(size[0],size[1],size[2],1)).astype(int)
    #    model_data2=np.resize(model_data2,(size[0],size[1],size[2],1)).astype(int)
    #return ([model_data1[match[i*(N//100)][0]] for i in range(0,100)], [model_data2[match[i*(N//100)][1]] for i in range(0,100)])
    return (pict1, pict2)

def hist(matching_file, outdir):
    plt.close('all')
    match=read_mathcing(matching_file)
    scores=[(float(match[i][2]),i) for i in range(len(match))]
    scores.sort()
    score=[float(match[int(i)][2]) for _,i in scores]
    plt.hist(score, 50, density=True, facecolor='g', alpha=0.75)
    plt.savefig('{}/hist_{}.png'.format(outdir, 1+max([m[0] for m in match])))
    plt.show()
    plt.close()

def hist_radius(model_path1, model_path2, matching_file, outdir):
    plt.close('all')
    match=read_mathcing(matching_file)
    N=len(match)
        
    model_data1=read_model_data(model_path1, N)
    model_data2=read_model_data(model_path2, N)

    model_data1 = color_resize(model_data1)
    model_data2 = color_resize(model_data2)

    pict1=[model_data1[match[i][0]] for i in range(N)]
    pict2=[model_data2[match[i][1]] for i in range(N)]

    est1 = estimate_radius(pict1)
    est2 = estimate_radius(pict2)
        
    print("Done hist_radius")
    plt.hist([est1, est2], label=["Train", model_path2.split('/')[2]])
    plt.legend()
    plt.savefig('{}/hist_radius.png'.format(outdir))
    plt.show()
    plt.close()
    
def color_resize(model_data):
    size=np.shape(model_data)
    if(size[1]!=size[2]):
        print("www", size)
        model_data=np.resize(model_data,(size[0],size[1],size[2]//3,3)).astype(int)
    return model_data
                        
def plot_matching_pairs(model_path1, model_path2, matching_file, N, outdir):
    model_data1=read_model_data(model_path1, N)
    model_data2=read_model_data(model_path2, N)

    model_data1 = color_resize(model_data1)
    model_data2 = color_resize(model_data2)
        
    match=read_mathcing(matching_file)

    # ===== Print first N pair of matching ====
    N=100
    batch=str(len(model_data1))
    pict1=[model_data1[match[i][0]] for i in range(N)]
    pict2=[model_data2[match[i][1]] for i in range(N)]
    score=[match[i][2] for i in range(N)]

    print(np.shape(pict1), np.shape(pict2))
    print_pictures(pict1,pict2,score,"random_"+batch, outdir)
    # ===== Print N best picture =====
    scores=[(float(match[i][2]),i) for i in range(len(match))]
    scores.sort()
    scores=scores[:N]
    pict1=[model_data1[match[int(i)][0]] for _,i in scores]
    pict2=[model_data2[match[int(i)][1]] for _,i in scores]
    score=[match[int(i)][2] for _,i in scores]
    print_pictures(pict1,pict2,score,"best_"+batch, outdir)
    #print(model_data1[0])

    # ===== Print first N pair of matching ====
    scores=[(float(match[i][2]),i) for i in range(len(match))]
    scores.sort()
    scores=scores[-N:]
    pict1=[model_data1[match[int(i)][0]] for _,i in scores]
    pict2=[model_data2[match[int(i)][1]] for _,i in scores]
    score=[match[int(i)][2] for _,i in scores]
    print_pictures(pict1,pict2,score,"worst_"+batch, outdir)

    scores=[float(match[i][2]) for i in range(len(match))]
    scores.sort()
    N=int(len(scores)*0.75)
    print("", np.mean(np.array(scores[0:N])))

def wgan_wgan_gp_type(N,type,second=False):
    title="WGAN, WGAN-GP Párosítás-Pontszám {} EPOCH".format(N)
    labels=["test","wgan","wgan-gp"]
    different_model_compare( files=["data/mnist/test/{}.txt".format(type),
            "models/wgan/generator_{}/{}.txt".format(N,type),
            "models/wgan-gp/generator_{}/{}.txt".format(N,type)],
            title=title,
            labels=labels,second=second)

def model_type(N,model,type,second=False):
    title="{} {}".format(model,type)
    labels=["test",model+" 1000",model+" 5000", model+" 10000"]
    different_model_compare( files=["data/mnist/test/{}.txt".format(type),
            "models/{}/generator_{}/{}.txt".format(model,1000,type),
            "models/{}/generator_{}/{}.txt".format(model,5000,type),
            "models/{}/generator_{}/{}.txt".format(model,10000,type)],
            title=title,
            labels=labels,second=second)

def new_model_type(N,model,type,second=False):
    title="WGAN, WGAN-GP Párosítás-Pontszám {} EPOCH".format(N)
    labels=["test",model+" 1000",model+" 5000", model+" 10000"]
    different_model_compare( files=["data/mnist/test/{}.txt".format(type),
            "models/{}/{}.txt".format(model,type)],
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
def celeba_example():
    # ===== Origin =====
    file="/home/doma/model_celeba10000.npy"
    data=np.load(file)
    plt.imshow(data[0])
    plt.show()

    # ===== Reconstruct =====
    model_data1=read_model_data("models/celeba/train/data")
    model_data2=[]
    for img in model_data1:
        model_data2.append(np.resize(img,(64,64,3)))
    model_data2=np.array(model_data2).astype(int)
    plt.imshow(model_data2[0].astype(int))
    plt.show()

import argparse
def parse():
    parser = argparse.ArgumentParser(description='Plot for specific model')
    parser.add_argument('-dataset', metavar='STRING', dest="dataset",
                    type = str, help='The dataset name', required=True)
    parser.add_argument('-N', metavar='INT', dest="N", type = int,
                    help='Number of max batch size', default = 2000)
    parser.add_argument('-r', metavar='INT', dest="r", type = int,
                    help='Range of batch steps (r, N)', default = 100)
    parser.add_argument('-folder', metavar='FILE', dest="folder", type = str,
                    help='The folder, where you want the output', default = ".")
    parser.add_argument('-choose', metavar='STR', dest="mode", type=str,
                    help="Mode foor different type of data")
    parser.add_argument('-batchs', metavar='TUPLE', dest="batchs", type=str,nargs="+",
                    help="Batchs example: generator_8000")
    parser.add_argument('-size', metavar='TUPLE', dest="size", type=int,nargs="+",
                    help="Size example: 64 64 1")
    parser.add_argument('-batchsize', metavar='INT', dest="batchsize", type=int, default=-1,
                                            help="Size of batch for model_progress")
    return parser
    
if(__name__=="__main__"):
    # ===== INIT models =====
    args=parse().parse_args()
    train_folder = "models/{}/train".format(args.dataset)
    test_folder = "models/{}/test".format(args.dataset)
    outdir = "models/{}".format(args.dataset)
    gen_models = ["wgan", "wgan-gp"]
    all_models = [test_folder]

    print(args.batchs)
    models_with_batch=[]
    for model in gen_models:
        model_with_batch =[]
        for batch in args.batchs:
            # ===== Generate modell, and evaluate =====
            act_model = "models/{}/{}/{}".format(args.dataset, model, batch)
            all_models.append(act_model)
            model_with_batch.append(act_model)
        models_with_batch.append(model_with_batch)

    # ===== CHOOSE mode
    if(args.mode ==  "epochs"):
        batch = args.batchsize
        files=gen_data_progress_of_model(models_with_batch, "compare.txt", batch)
        different_model_compare(files,
            title="Progress of specific models {}".format(args.dataset),
                                labels=gen_models, outfile=outdir+"/model_progress_{}.png".format(batch))
    elif(args.mode == "radius"):
        for act_model in all_models:
            hist_radius(train_folder+"/data", act_model+"/data",
                        act_model+"/mnist_result_{}.txt".format(1000), act_model)
        
    elif(args.mode=="test"):
        different_model_compare(files=[
            "models/celeba/test/compare.txt", 
            "models/celeba/test/compare.txt.log"],
                title="matching celeba",
                labels=["test", "time"],time=True)
        
    elif(args.mode =="pict"):
        model1="models/trash/celeba/train/"#"data/mnist/train/"
        model2="models/trash/celeba/dani/"#"models/wgan/generator_1000"
        plot_matching_pairs(model1+"data",model2+"/data",
                            model2+"/mnist_result_{}.txt".format(1000), 1000 , 'models')
        
    exit()
