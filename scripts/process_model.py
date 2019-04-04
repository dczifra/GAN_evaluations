from keras.models import load_model
from keras.models import model_from_json
from eval.try_eval import fid_score
import numpy as np
import os
import sys
import time

class Models:
    nojson=False
    myTimer=None
    N=2500
    range=50
    log=False

    def get_model(model_file):
        if(Models.nojson):
            from keras.models import load_model
            model = load_model(model_file+".h5")
            return model
        # load json and create model
        json_file = open(model_file+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(model_file+".h5")
        print("Loaded model ({}) from disk".format(model_file))
        return loaded_model

    def transform_num(elem,npy=False):
        if(npy):
            return str(int(128.0*(1.0+elem)))
        else:
            return str(int(128.0*(1.0+elem[0])))

    def generate_samples(model,output_file,N=1000):
        # ===== Generate samples: =====
        batch_size=N
        latent_dim=100
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Generate a batch of new images
        gen_imgs = model.predict(noise)
        #print(np.shape(gen_imgs))

        # ===== Write out to file =====
        if(not os.path.isdir(output_file)):
            os.makedirs(output_file)
        iter=0
        for img in gen_imgs:
            myfile=open(output_file+"/image_"+str(iter)+".txt","w")
            for row in img:
                for elem in row:
                    myfile.write(Models.transform_num(elem)+" ")
                myfile.write("\n")
            
            iter+=1
            if(iter >N): break;
            elif(Models.log): print("\r{}".format(iter),end=" ")
            myfile.close()

    def generate_from_npy(input,output_file):
        N=Models.N
        print("Loading {} ...".format(input))
        data=np.load(input)
        print("Loaded")
        # ===== Write out to file =====
        if(not os.path.isdir(output_file)):
            os.makedirs(output_file)

        size=np.shape(data)
        print(size)
        data=np.resize(data,(size[0],size[1],size[2]*size[3]))
        noTransform=(np.max(data[0])>2)
        iter=0
        for img in data:
            myfile=open(output_file+"/image_"+str(iter)+".txt","w")
            for row in img:
                for elem in row:
                    if(noTransform):
                        myfile.write(str(elem)+" ")
                    else:
                        myfile.write(Models.transform_num(elem,True)+" ")
                myfile.write("\n")
            iter+=1
            if(iter >N): break;
            elif(Models.log): print("\r{}".format(iter),end=" ")
            myfile.close()

    def generate_mnist(N,test=False,):
        # ===== Get MNIST =====
        from keras.datasets import mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        print("Size of MNIST: ",len(x_train),len(x_test))

        # ===== Choose dataset, and create folder(s) =====
        filename="data/mnist/"+("test/data" if test else "train/data")
        data=(x_test if test else x_train)
        np.random.shuffle(data)
        if(not os.path.isdir(filename)):
            os.makedirs(filename)
            
        for i in range(min(N,len(data))):
            myfile=open(filename+"/image_"+str(i)+".txt","w")
            for row in data[i]:
                for elem in row:
                    myfile.write(str(elem)+" ")
                myfile.write("\n")

    def tutorial():
        Models.generate_mnist(10,False)
        gen_model=Models.get_model("data/mnist_wgan/generator_1000")
        Models.generate_samples(gen_model,"data/mnist/wgan")

    def stretching_limits(n,r,model_filename,gen=True):
        #model_filename="models/wgan-gp/generator_1000"
        sample_size=n
        if(gen):
            gen_model=Models.get_model(model_filename)
            Models.generate_samples(gen_model,model_filename+"/data",sample_size)

        #for N in range(r,n+r,r):
        cmd="bin/main -size 28,28 -folder1 data/mnist/train/data -folder2 {}/data -N {} -range {} -out {}/compare_limit.txt".format(model_filename,n,r,model_filename)
        print(cmd)
        Models.measure_process(cmd,"Hun",n,"")

            #cmd="bin/main -size 28,28 -folder1 data/mnist/train/data -folder2 models/wgan-gp/generator_1000/data -N {} -range {} -out models/wgan-gp/generator_1000/compare_limit.txt -flow".format(N,N)
            #print(cmd)
            #Models.measure_process(cmd,"Flow",N,"")
    
    def process_modell(model_filename,sample_size=10000,generate=False):
        if(generate):
            gen_model=Models.get_model(model_filename)
            Models.generate_samples(gen_model,model_filename+"/data",sample_size)

        N=Models.N
        r=Models.range
        args=["bin/main",
            "-size","28,28",
            "-folder1","data/mnist/train/data",\
            "-folder2",model_filename+"/data","-N {}".format(N),"-range {}".format(r),\
            "-out"]

        print(" ".join(args+[model_filename+"/compare.txt"]))
        Models.myTimer.write("====== {} ======\n".format(model_filename))
        Models.measure_process(" ".join(args+[model_filename+"/compare.txt"]),"Hun",N,r)
        Models.measure_process(" ".join(args+[model_filename+"/flow.txt","-flow"]),"Flow",N,r)
        Models.measure_process(" ".join(args+[model_filename+"/deficit.txt","-deficit"]),"Deficit",N,r)
        Models.measure_process(" ".join(args+[model_filename+"/defFlow.txt","-defFlow"]),"Deflow",N,r)
        #fid_score("data/mnist/train",model_filename)

    def process_celeba(model_filename,generate=None):
        if(generate!=None):
            Models.generate_from_npy(generate,model_filename+"/data")
        N=Models.N
        r=Models.range
        args=["bin/main",
            "-size","64,192",
            "-folder1","models/celeba/train/data",\
            "-folder2",model_filename+"/data","-N {}".format(N),"-range {}".format(r),\
            "-out"]

        print(" ".join(args+[model_filename+"/compare.txt"]))
        os.system(" ".join(args+[model_filename+"/compare.txt"]))
    def measure_process(command, type_,N,r):
        start=time.time()
        os.system(command)
        p1=time.time()
        Models.myTimer.write("{} with N={} range={} : {} \n".format(type_,N,r,p1-start))

    def run_all(gen=False):
        Models.nojson=False
        Models.process_modell("data/mnist/test")
        for model in [1000,5000,10000]:
            Models.process_modell("models/wgan/generator_{}".format(model),generate=gen)
            Models.process_modell("models/wgan-gp/generator_{}".format(model),generate=gen)
            exit()

    def new_models():
        gen=True
        Models.nojson=True
        #,"cgan_mnist","
        for model in ["dcgan_mnist","lsgan_mnist","wgan_mnist"]:
            Models.process_modell("models/downloaded/{}".format(model),generate=gen)


if(__name__=="__main__"):
    Models.N=int(sys.argv[2])
    Models.range=int(sys.argv[3])
    Models.myTimer=open("mytimer.txt","w")

    if(sys.argv[1]=="init"):
        #Models.generate_from_npy("/home/doma/celeba_64_64_color.npy","models/celeba/train/data")
        Models.generate_from_npy("/home/datasets/celeba_64_64_color.npy","models/celeba/train/data")
    if(sys.argv[1]=="limits"):
        Models.stretching_limits(Models.N,Models.range,"data/mnist/test",gen=False)
        Models.stretching_limits(Models.N,Models.range,"models/wgan-gp/generator_10000")
        Models.stretching_limits(Models.N,Models.range,"models/wgan/generator_10000")
    elif(sys.argv[1]=="celeba"):
        #Models.log=True
        #Models.process_celeba("models/celeba/test","/home/doma/model_celeba10000.npy")
        Models.process_celeba("models/celeba/gen_9999","/home/zombori/wgan_gp_orig/generated/celeba_9999.npy")
        Models.process_celeba("models/celeba/gen_49999","/home/zombori/wgan_gp_orig/generated/celeba_49999.npy")        
        Models.process_celeba("models/celeba/gen_99999","/home/zombori/wgan_gp_orig/generated/celeba_99999.npy")
        Models.process_celeba("models/celeba/gen_149999","/home/zombori/wgan_gp_orig/generated/celeba_149999.npy")
    Models.myTimer.close()
    exit()

    
    Models.run_all(sys.argv[1]=="True")
    

    Models.myTimer=open("limits.txt","w")
    Models.stretching_limits()
    Models.myTimer.close()

    exit()
    
    

