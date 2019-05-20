import matplotlib.pyplot as plt
import numpy as np
import sys

def different_model_compare( files,title,labels,
                             xlabel="Minta meret",ylabel="[Párosítás]/[Minta meret]",second=False, time=False, outfile="."):

    colors=['r','g','b','y','p','r--']
    fig, ax1 = plt.subplots()
    if(second): 
        ax2=ax1.twinx()
        ax2.set_ylabel("Value of the flow/Max matching")

    i=0
    for file in files:
        test=open(file)
        a,b,delta=np.array(test.readline().split(" ")).astype(np.int)
        data=np.array(test.readline().split(" "))[:-1].astype(np.float)
        
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


from PIL import Image
def plotImages(data, n_x, n_y, name, text=None):
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
    
def read_model_data(model_path,N,mod2="image"):
    model_data=[]
    for i in range(N):
        myfile=open(model_path+"/{}_{}.txt".format(mod2,i))
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

def print_pictures(pict1, pict2, score,ind, outdir):
    plt.close('all')
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
    plt.savefig('{}/matching{}.png'.format(outdir, ind))
    plt.show()

def get_nth_matching(model_path1, model_path2, matching_file):
    match=read_mathcing(matching_file)
    N=len(match)
    model_data1=read_model_data(model_path1, N)
    model_data2=read_model_data(model_path2, N)
    size=np.shape(model_data1)
    if(size[-1]!=size[-2]):
        #print(size)
        model_data1=np.resize(model_data1,(size[0],size[1],size[2]//3,3)).astype(int)
        model_data2=np.resize(model_data2,(size[0],size[1],size[2]//3,3)).astype(int)
    else:
        model_data1=np.resize(model_data1,(size[0],size[1],size[2],1)).astype(int)
        model_data2=np.resize(model_data2,(size[0],size[1],size[2],1)).astype(int)
    return ([model_data1[match[i*(N//100)][0]] for i in range(0,100)], [model_data2[match[i*(N//100)][1]] for i in range(0,100)])

def hist(matching_file, outdir):
    plt.close('all')
    match=read_mathcing(matching_file)
    scores=[(float(match[i][2]),i) for i in range(len(match))]
    scores.sort()
    score=[float(match[int(i)][2]) for _,i in scores]
    plt.hist(score, 50, density=True, facecolor='g', alpha=0.75)
    #plt.axis([40, 1000, 0, 0.003])
    plt.savefig('{}/hist_{}.png'.format(outdir, 1+max([m[0] for m in match])))
    plt.show()
    plt.close()
    
def plot_matching_pairs(model_path1, model_path2, matching_file, N, outdir):
    model_data1=read_model_data(model_path1, N)
    model_data2=read_model_data(model_path2, N)

    size=np.shape(model_data1)
    if(size[-1]!=size[-2]):
        print(size)
        model_data1=np.resize(model_data1,(size[0],size[1],size[2]//3,3)).astype(int)
        model_data2=np.resize(model_data2,(size[0],size[1],size[2]//3,3)).astype(int)
    else:
        model_data1=np.resize(model_data1,(size[0],size[1],size[2])).astype(int)
        model_data2=np.resize(model_data2,(size[0],size[1],size[2])).astype(int)
        
    match=read_mathcing(matching_file)

    # ===== Print first N pair of matching ====
    N=20
    batch=str(len(model_data1))
    pict1=[model_data1[match[i][0]] for i in range(N)]
    pict2=[model_data2[match[i][1]] for i in range(N)]
    score=[match[i][2] for i in range(N)]

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

if(__name__=="__main__"):
# Param1: N --> model epoch (Pl.: 1000, 5000, 10000)
# Param3: batch size: number of samples from the dataset
    
    if(sys.argv[1]=="toy"):
        model=sys.argv[2] 
        different_model_compare(files=[
            "models/{}/test/compare.txt".format(model),
            "models/{}/wgan/gen_8000/generator_8000/compare.txt".format(model),
            "models/{}/wgan-gp/gen_8000/generator_8000/compare.txt".format(model)],
                                title="Matchig loss {}".format(model),
                                labels=["test", "wgan", "wgan-gp"],
                                outdir = "models/{}".format(model))
    elif(sys.argv[1]=="toypict"):
        model=sys.argv[2]
        generator = sys.argv[3]
        plot_hist = (sys.argv[4] == "True")
        model1="models/{}/train/".format(model)
        model2="models/{}/{}".format(model, generator)
        outdir="models/{}".format(model)
        if(plot_hist):
            hist(model1+"data",model2+"/data", model2+"/mnist_result_{}.txt".format(1000), outdir)
        else:
            plot_matching_pairs(model1+"data",model2+"/data", model2+"/mnist_result_{}.txt".format(1000),1000, outdir)
        
        
                                
    elif(sys.argv[1]=="test"):
        different_model_compare(files=[
            "models/celeba/test/compare.txt", 
            "models/celeba/test/compare.txt.log"],
                title="matching celeba",
                labels=["test", "time"],time=True)
    elif(sys.argv[1]=="matching"):
        different_model_compare(files=["models/celeba/gen_9999/compare.txt",
        "models/celeba/gen_99999/compare.txt",
        "models/celeba/gen_149999/compare.txt",
        "models/celeba/dani/compare.txt","models/celeba/test/compare.txt"],
                title="matching celeba",
                labels=["celeba 9999","99999","149999","Dani","test"],second=False)
        
    elif(sys.argv[1]=="pict"):
        model1="models/trash/celeba/train/"#"data/mnist/train/"
        model2="models/trash/celeba/dani/"#"models/wgan/generator_1000"
        plot_matching_pairs(model1+"data",model2+"/data",
                            model2+"/mnist_result_{}.txt".format(1000), 1000 , 'models')
    else:
        different_model_compare(files=["models/wgan-gp/compare_10000.txt",
        "models/wgan/generator_1000/compare.txt",
        "models/wgan-gp/generator_10000/compare_limit.txt",
        "models/wgan/generator_10000/compare_limit.txt",
        "data/mnist/test/compare_limit.txt"],
                title="matching mnist until 10 000",
                labels=["wgan-gp 1000","wgan 1000","wgan-gp 10 000","wgan 10 000", "test"],second=False)
    exit()


    #FID_for_MNIST()
    # TODO: Parameter descriptions!!!!
    if(len(sys.argv)>2):
        #batch=250
        batch=int(sys.argv[2])
        plot_matching_pairs("data/mnist/train/data","models/wgan/generator_1000/data",
            "models/wgan/generator_1000/mnist_result_{}.txt".format(batch))
        plot_matching_pairs("data/mnist/train/data","models/wgan-gp/generator_1000/data",
            "models/wgan-gp/generator_1000/mnist_result_{}.txt".format(batch))
        plot_matching_pairs("data/mnist/train/data","data/mnist/test/data",
            "data/mnist/test/mnist_result_{}.txt".format(batch))
        N=int(sys.argv[1])
        exit()
    elif(len(sys.argv)>1):
        N=int(sys.argv[1])
    else: N=1000

    models=["wgan","wgan-gp","downloaded/lsgan_mnist","downloaded/wgan_mnist"]
    for type in ["compare","flow","deficit","defFlow"]:#,"fid_score"]:
        #new_model_type(N,models[2],type,type=="defFlow")
        #model_type(N,models[0],type,type=="defFlow")
        #model_type(N,models[1],type,type=="defFlow")
        wgan_wgan_gp_type(N,type,type=="defFlow")
    exit()
