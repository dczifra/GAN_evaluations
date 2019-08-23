import numpy as np
import os
import sys
import time
import keras

from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model
from keras.models import model_from_json
from eval.try_eval import fid_score
from plots import read_model_data

import plots
import numbers
# ========================================
#              Helper Functions
# ========================================

def write_images(gen_imgs, folder):
    iter = 0
    for img in gen_imgs:
        myfile=open(folder+"/image_"+str(iter)+".txt","w")
        for row in img:
            for elem in row:
                myfile.write(transform_num(elem)+" ")
            myfile.write("\n")
        myfile.close()
        iter+=1

def transform_num(elem,npy=False):
    if(npy or isinstance(elem, numbers.Number)):
        return str(int(128.0*(1.0+elem)))
    else:
        return str(int(128.0*(1.0+elem[0])))

class Models:
    N=2500
    range=50
    log=False
    nojson=False
    myTimer=None

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

    # TDOD: 3 csatorna
    def generate_samples(model,output_file,N=1000):
        # ===== Generate samples: =====
        batch_size=N
        latent_dim=model.layers[0].input_shape[-1]
        print("Latent dim {}".format(latent_dim))
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Generate a batch of new images
        gen_imgs = model.predict(noise)

        # ===== Write out to file =====
        if(not os.path.isdir(output_file)):
            os.makedirs(output_file)
        
        write_images(gen_imgs[:N], output_file)

    def generate_from_npy(input, output_file, parity=None):
        N=Models.N
        print("Loading {} ...".format(input))

        data=np.load(input)
        size=np.shape(data)
        data=np.resize(data,(size[0],size[1],size[2]*size[3]))
        print(size)
        
        # ===== Write out to file =====
        if(not os.path.isdir(output_file)):
            os.makedirs(output_file)

        noTransform=(np.max(data[0])>2)
        iter=0
        iter2=0
        for img in data[:2*N]:
            # ===== Train and test =====
            if(parity == None or iter%2==parity):
                #print(iter)
                myfile=open(output_file+"/image_"+str(iter)+".txt","w")
                for row in img:
                    for elem in row:
                        if(noTransform):
                            myfile.write(str(elem)+" ")
                        else:
                            myfile.write(transform_num(elem,True)+" ")
                    myfile.write("\n")
                myfile.close()
                iter2+=1
            iter+=1
            if(iter2 > N): break;
            elif(Models.log): print("\r{}".format(iter),end=" ")
            
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

    def generate_feature(picture_folder, shape, feature_net ):
        """ Generates features from the given pictures """
        datas = read_model_data(picture_folder)
        datas = np.resize(datas, (-1, shape[0],shape[1],shape[2]))

        vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_tensor = Input(shape=shape), input_shape=None, pooling=None, classes=1000)
        all_layers_model = Model(inputs = vgg.input, outputs = [layer.output for layer in vgg.layers][1:])
        layers = all_layers_model.predict(datas[0:100])
        layers = [elem[0][0] for elem in layers]
        features = [[img] for img in layers[-1]]
        
        print("Shape of features: {}".format(np.shape(features)))

        new_folder = os.path.join( picture_folder, "../", feature_net)
        if(not os.path.isdir(new_folder)):
            os.makedirs(new_folder)
        write_images(features, new_folder)

        return np.shape(features)[1:]

    def toy_black_white(model_filename = "models/dataset/batch_name", dataset= "dsprite",
                        generate=False, size = [64,64,1], feature = False):
        if(generate):
            if(os.path.exists(model_filename+".npy")):
                Models.generate_from_npy(model_filename+".npy", model_filename+"/data")
            else:
                gen_model=Models.get_model(model_filename)
                Models.generate_samples(gen_model, model_filename+"/data",Models.N)

        network = "data"
        if(feature):
            network = "vgg"
            Models.generate_feature(model_filename+"/data", size, network)
            size = Models.generate_feature("models/{}/train/data".format(dataset), size, network)
        elif(len(size)>2):
            size[1]*=size[2]
            
        size = str(size[0])+","+str(size[1])
        print("New size is: {}".format(size))

        args=["bin/main",
              "-size",size,
              "-folder1","models/{}/train/{}".format(dataset, network),
              "-folder2",model_filename+"/{}".format(network),
              "-N {}".format(Models.N),
              "-range {}".format(Models.range),
              "-out"]
        print(" ".join(args+[model_filename+"/compare.txt"]))
        os.system(" ".join(args+[model_filename+"/compare.txt"]))
        #Models.measure_process(" ".join(args+[model_filename+"/deficit.txt","-deficit"]),"Deficit",Models.N,Models.range)
        #Models.measure_process(" ".join(args+[model_filename+"/defFlow.txt","-defFlow"]),"Deflow",Models.N,Models.range)
        #fid_score("models/{}/train".format(dataset), model_filename) 
        
def parse():
    parser = argparse.ArgumentParser(description='Plot for specific model')
    parser.add_argument('-dataset', metavar='STRING', dest="dataset",
        type = str, help='The dataset name', required=True)
    parser.add_argument('-N', metavar='INT', dest="N", type = int,
        help='Number of max batch size', default = 2000)
    parser.add_argument('-r', metavar='INT', dest="r", type = int,
        help='Range of batch steps (r, N)', default = 100)
    parser.add_argument('-folder', metavar='FILE', dest="folder", type = str,
        help='The follder, where you want the output', default = ".")
    parser.add_argument('-mode', metavar='STR', dest="mode", type=str,
                        help="Mode foor different type of data")
    parser.add_argument('-batchs', metavar='TUPLE', dest="batchs", type=str, nargs="+",
                help="Batchs example: generator_8000")
    parser.add_argument('-size', metavar='TUPLE', dest="size", type=int, nargs="+",
                                        help="Size example: 64 64 1")
    parser.add_argument('--plot', metavar='BOOLEAN', dest="plot", const = True, default = False, nargs='?')
    parser.add_argument('--featureDist', metavar='BOOLEAN', dest="featureDist", const = True, default = False, nargs='?')
    return parser

import argparse
if(__name__=="__main__"):
    args=parse().parse_args()
    Models.N=args.N
    Models.range=args.r
    Models.myTimer=open("mytimer.txt","w")

    if(args.mode == "feature"):
        print("Hello {}".format(args.featureDist))
        act_model = "models/{}/{}/{}".format(args.dataset, "wgan", "generator_1000")
        Models.toy_black_white(act_model, args.dataset, False, args.size, args.featureDist)
        #size alallit

    elif(args.mode == "toy"):
        train_folder = "models/{}/train".format(args.dataset)
        test_folder = "models/{}/test".format(args.dataset)
        new_test_folder = "models/{}/new_test".format(args.dataset)
        outdir = "models/{}".format(args.dataset)
        print(args.plot)
        if(not args.plot):
            Models.generate_from_npy(train_folder+".npy", train_folder+"/data")
            Models.generate_from_npy(test_folder+".npy", test_folder+"/data")
            #Models.generate_from_npy(new_test_folder+".npy", new_test_folder+"/data")

        
        #gen_models = ["good_wgan", "wgan", "wgan-gp"]
        gen_models = ["wgan", "wgan-gp"]  
        all_models = [test_folder]
        labels = ["test"]+ gen_models
        for batch in args.batchs:
            for model in gen_models:
                # ===== Generate modell, and evaluate =====
                act_model = "models/{}/{}/{}".format(args.dataset, model, batch)
                all_models.append(act_model)
        for act_model in all_models:
            if(not args.plot):
                Models.toy_black_white(act_model, args.dataset, ("test" not in act_model), args.size)
            # ===== Plot Matching =====
            for r in range(Models.range, Models.N+Models.range, Models.range):
                print('\r',r, end='')
                #plots.plot_matching_pairs(train_folder+"/data", act_model+"/data",
                #    act_model+"/mnist_result_{}.txt".format(r),r,act_model)
                #plots.hist(act_model+"/mnist_result_{}.txt".format(r), act_model)
                
            (m1,m2) = plots.get_nth_matching(train_folder+"/data",act_model+"/data", act_model+"/mnist_result_{}.txt".format(Models.N))
            plots.plotImages(m1, 10, 10, act_model+"/every100_train.png", text=None)
            plots.plotImages(m2, 10, 10, act_model+"/every100_test.png", text=None)
            plots.hist_radius(train_folder+"/data", act_model+"/data",
                act_model+"/mnist_result_{}.txt".format(Models.N), act_model)
            plots.plot_matching_pairs(train_folder+"/data", act_model+"/data",
                act_model+"/mnist_result_{}.txt".format(Models.N),r,act_model)
            plots.hist(act_model+"/mnist_result_{}.txt".format(Models.N), act_model) 
        # ===== Compare models =====
        plots.different_model_compare(files=[model+"/compare.txt" for model in all_models],
            title="Párosítás Pontszám {}".format(args.dataset),
            labels=labels, outfile=outdir+"/model_compare.png")
        plots.different_model_compare(files=[model+"/deficit.txt" for model in all_models],
            title="Hiány {}".format(args.dataset.upper()),
                                      labels=labels, xlabel="Elhagyott élek száma", ylabel = "Maximális Párosítás mérete (%)", outfile=outdir+"/deficit.png")
        plots.different_model_compare(files=[model+"/defFlow.txt" for model in all_models],
            title="Átlagos folyamérték {}".format(args.dataset.upper()),
            xlabel="Elhagyott élek száma", ylabel="[Párosítás érték]/[Párosítás méret]", labels=labels, outfile=outdir+"/deFlow.png")
        
    elif(args.mode=="one"):
        Models.process_celeba("models/celeba/test",generate=None)
    elif(args.mode=="init"):
        #Models.generate_from_npy("/home/doma/celeba_64_64_color.npy","models/celeba/train/data")
        Models.generate_from_npy("/home/datasets/celeba_64_64_color.npy","models/celeba/train/data",0)
        Models.generate_from_npy("/home/datasets/celeba_64_64_color.npy","models/celeba/test/data",1)
        Models.process_celeba("models/celeba/test",generate=None)
        print("=== End of init ===")
    elif(args.mode=="limits"):
        Models.stretching_limits(Models.N,Models.range,"data/mnist/test",gen=False)
        Models.stretching_limits(Models.N,Models.range,"models/wgan-gp/generator_10000")
        Models.stretching_limits(Models.N,Models.range,"models/wgan/generator_10000")
    elif(args.mode=="celeba"):
        #Models.log=True
        #Models.process_celeba("models/celeba/test","/home/doma/model_celeba10000.npy")
        Models.process_celeba("models/celeba/dani","/mnt/g2home/daniel/experiments/vae/inverting-is-hard/stylegan-fork/saved/generated.npy")
        Models.process_celeba("models/celeba/test",generate=None)
        for epoch in range(9999,190000,10000):
            Models.process_celeba("models/celeba/gen_{}".format(epoch),
            "/mnt/g2home/zombori/wgan_gp_orig/generated/celeba_{}.npy".format(epoch))
    
    Models.myTimer.close()
    exit()
