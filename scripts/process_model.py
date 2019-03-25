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

    def transform_num(elem):
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

    def stretching_limits():
        model_filename="models/wgan-gp/generator_1000"
        sample_size=10000
        gen_model=Models.get_model(model_filename)
        Models.generate_samples(gen_model,model_filename+"/data",sample_size)

        for N in range(500,10001,500):
            cmd="bin/main -size 28,28 -folder1 data/mnist/train/data -folder2 models/wgan-gp/generator_1000/data -N {} -range {} -out models/wgan-gp/generator_1000/compare_limit.txt".format(N,N)
            print(cmd)
            Models.measure_process(cmd,"Hun",N,"")

            cmd="bin/main -size 28,28 -folder1 data/mnist/train/data -folder2 models/wgan-gp/generator_1000/data -N {} -range {} -out models/wgan-gp/generator_1000/compare_limit.txt -flow".format(N,N)
            print(cmd)
            Models.measure_process(cmd,"Flow",N,"")
    
    def process_modell(model_filename,sample_size=2000,generate=False):
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
    Models.N=sys.argv[2]
    Models.range=sys.argv[3]
    Models.myTimer=open("mytimer.txt","w")
    Models.run_all(sys.argv[1]=="True")
    Models.myTimer.close()

    Models.myTimer=open("limits.txt","w")
    Models.stretching_limits()
    Models.myTimer.close()

    exit()

    #Models.new_models()

    gen=('True'==sys.argv[1])
    print(gen)
    Models.process_modell("data/mnist/test")
    
    

