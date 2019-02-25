from keras.models import load_model
from keras.models import model_from_json
import numpy as np
import os

def get_model(model_file):
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
                myfile.write(transform_num(elem)+" ")
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
        myfile=open(filename+"/number_"+str(i)+".txt","w")
        for row in data[i]:
            for elem in row:
                myfile.write(str(elem)+" ")
            myfile.write("\n")

def tutorial():
    generate_mnist(10,False)
    gen_model=get_model("data/mnist_wgan/generator_1000")
    generate_samples(gen_model,"data/mnist/wgan")

def process_modell(model_filename,sample_size=2000,generate=False):
    if(generate):
        gen_model=get_model(model_filename)
        generate_samples(gen_model,model_filename+"/data",sample_size)

    args=["bin/main",
        "-size","28,28",
        "-folder1","data/mnist/train/data",\
        "-folder2",model_filename+"/data","-N 2000","-range 100",\
        "-out"]

    os.system(" ".join(args+[model_filename+"/compare.txt"]))
    print(" ".join(args+[model_filename+"/compare.txt"]))
    os.system(" ".join(args+[model_filename+"/flow.txt","-flow"]))
    os.system(" ".join(args+[model_filename+"/deficit.txt","-deficit"]))
    #os.system(" ")



if(__name__=="__main__"):
    process_modell("data/mnist/test")
    exit(1)
    process_modell("models/wgan/generator_1000",generate=True)
    process_modell("models/wgan-gp/generator_1000",generate=True)
    

